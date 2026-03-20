use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

const MODEL_INPUT_SIZE: u32 = 640;
const CONF_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.4;
const INPUT_MEAN: f32 = 127.5;
const INPUT_STD: f32 = 128.0;

const STRIDES: [u32; 3] = [8, 16, 32];
const NUM_ANCHORS: u32 = 2;

#[derive(Debug, Clone, Copy)]
pub struct FaceLandmark {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: [f32; 4], // x1, y1, x2, y2
    pub confidence: f32,
    /// 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    pub landmarks: [FaceLandmark; 5],
}

pub struct FaceDetector {
    session: Session,
}

impl FaceDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .context("Failed to create session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to set optimization level: {}", e);
                e.recover()
            })
            .with_execution_providers([ep::CUDA::default().build()])
            .unwrap_or_else(|e| {
                tracing::warn!("CUDA EP not available, falling back to CPU: {}", e);
                e.recover()
            })
            .commit_from_file(model_path)
            .context(format!("Failed to load model from {}", model_path))?;

        tracing::info!("Face detection model loaded from {}", model_path);
        tracing::info!(
            "Model has {} outputs",
            session.outputs().len()
        );

        Ok(FaceDetector { session })
    }

    /// Run face detection on an RGB image.
    pub fn detect_rgb(&mut self, image: &RgbImage) -> Result<Vec<FaceDetection>> {
        let (input_data, scale, pad_x, pad_y) = preprocess_rgb(image);
        self.detect_raw(image.width(), image.height(), input_data, scale, pad_x, pad_y)
    }

    /// Run face detection on a grayscale (IR) image.
    pub fn detect_gray(&mut self, image: &GrayImage) -> Result<Vec<FaceDetection>> {
        let (input_data, scale, pad_x, pad_y) = preprocess_gray(image);
        self.detect_raw(image.width(), image.height(), input_data, scale, pad_x, pad_y)
    }

    fn detect_raw(
        &mut self,
        img_w: u32,
        img_h: u32,
        input_data: Vec<f32>,
        scale: f32,
        pad_x: f32,
        pad_y: f32,
    ) -> Result<Vec<FaceDetection>> {

        let input_tensor = Tensor::from_array((
            [1usize, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize],
            input_data.into_boxed_slice(),
        ))
        .context("Failed to create input tensor")?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .context("Inference failed")?;

        let num_outputs = outputs.len();
        let fmc = if num_outputs == 9 || num_outputs == 6 {
            3
        } else {
            tracing::warn!("Unexpected number of outputs: {}, assuming fmc=3", num_outputs);
            3
        };
        let has_kps = num_outputs >= 9;

        let mut detections = Vec::new();

        for (idx, &stride) in STRIDES.iter().enumerate().take(fmc) {
            let feat_h = MODEL_INPUT_SIZE / stride;
            let feat_w = MODEL_INPUT_SIZE / stride;

            // Extract score, bbox, and kps tensors for this stride
            let (_, scores_data) = outputs[idx]
                .try_extract_tensor::<f32>()
                .context("Failed to extract score tensor")?;
            let (_, bbox_data) = outputs[idx + fmc]
                .try_extract_tensor::<f32>()
                .context("Failed to extract bbox tensor")?;
            let kps_data = if has_kps {
                let (_, data) = outputs[idx + fmc * 2]
                    .try_extract_tensor::<f32>()
                    .context("Failed to extract kps tensor")?;
                Some(data)
            } else {
                None
            };

            // Generate anchors and decode detections
            let num_positions = (feat_h * feat_w) as usize;

            for pos in 0..num_positions {
                let grid_y = pos as u32 / feat_w;
                let grid_x = pos as u32 % feat_w;
                let anchor_x = (grid_x as f32) * stride as f32;
                let anchor_y = (grid_y as f32) * stride as f32;

                for anchor in 0..NUM_ANCHORS {
                    let anchor_idx = pos * NUM_ANCHORS as usize + anchor as usize;

                    let score = scores_data[anchor_idx];
                    if score < CONF_THRESHOLD {
                        continue;
                    }

                    // Decode bbox: distance from anchor, scaled by stride
                    let bbox_offset = anchor_idx * 4;
                    let d0 = bbox_data[bbox_offset] * stride as f32;
                    let d1 = bbox_data[bbox_offset + 1] * stride as f32;
                    let d2 = bbox_data[bbox_offset + 2] * stride as f32;
                    let d3 = bbox_data[bbox_offset + 3] * stride as f32;

                    // Map back to original image coordinates
                    let x1 = ((anchor_x - d0 - pad_x) / scale).clamp(0.0, img_w as f32);
                    let y1 = ((anchor_y - d1 - pad_y) / scale).clamp(0.0, img_h as f32);
                    let x2 = ((anchor_x + d2 - pad_x) / scale).clamp(0.0, img_w as f32);
                    let y2 = ((anchor_y + d3 - pad_y) / scale).clamp(0.0, img_h as f32);

                    // Decode landmarks
                    let landmarks = if let Some(ref kps) = kps_data {
                        let kps_offset = anchor_idx * 10;
                        let mut lms = [FaceLandmark { x: 0.0, y: 0.0 }; 5];
                        for i in 0..5 {
                            let lx = anchor_x + kps[kps_offset + i * 2] * stride as f32;
                            let ly = anchor_y + kps[kps_offset + i * 2 + 1] * stride as f32;
                            lms[i] = FaceLandmark {
                                x: ((lx - pad_x) / scale).clamp(0.0, img_w as f32),
                                y: ((ly - pad_y) / scale).clamp(0.0, img_h as f32),
                            };
                        }
                        lms
                    } else {
                        [FaceLandmark { x: 0.0, y: 0.0 }; 5]
                    };

                    detections.push(FaceDetection {
                        bbox: [x1, y1, x2, y2],
                        confidence: score,
                        landmarks,
                    });
                }
            }
        }

        nms(&mut detections);

        Ok(detections)
    }
}

/// Preprocess RGB image: letterbox resize to 640x640, normalize with mean/std.
fn preprocess_rgb(image: &RgbImage) -> (Vec<f32>, f32, f32, f32) {
    let (img_w, img_h) = (image.width() as f32, image.height() as f32);
    let target = MODEL_INPUT_SIZE as f32;

    let scale = (target / img_w).min(target / img_h);
    let new_w = (img_w * scale) as u32;
    let new_h = (img_h * scale) as u32;

    let resized =
        image::imageops::resize(image, new_w, new_h, image::imageops::FilterType::Triangle);

    let pad_x = (MODEL_INPUT_SIZE - new_w) as f32 / 2.0;
    let pad_y = (MODEL_INPUT_SIZE - new_h) as f32 / 2.0;
    let pad_x_i = pad_x as u32;
    let pad_y_i = pad_y as u32;

    let mut padded = RgbImage::from_pixel(
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
        image::Rgb([0, 0, 0]),
    );
    image::imageops::overlay(&mut padded, &resized, pad_x_i as i64, pad_y_i as i64);

    let sz = MODEL_INPUT_SIZE as usize;
    let mut data = vec![0.0f32; 3 * sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let pixel = padded.get_pixel(x as u32, y as u32);
            data[0 * sz * sz + y * sz + x] = (pixel[0] as f32 - INPUT_MEAN) / INPUT_STD;
            data[1 * sz * sz + y * sz + x] = (pixel[1] as f32 - INPUT_MEAN) / INPUT_STD;
            data[2 * sz * sz + y * sz + x] = (pixel[2] as f32 - INPUT_MEAN) / INPUT_STD;
        }
    }

    (data, scale, pad_x, pad_y)
}

/// Preprocess grayscale image: letterbox resize to 640x640, replicate to 3 channels, normalize.
fn preprocess_gray(image: &GrayImage) -> (Vec<f32>, f32, f32, f32) {
    let (img_w, img_h) = (image.width() as f32, image.height() as f32);
    let target = MODEL_INPUT_SIZE as f32;

    let scale = (target / img_w).min(target / img_h);
    let new_w = (img_w * scale) as u32;
    let new_h = (img_h * scale) as u32;

    let resized =
        image::imageops::resize(image, new_w, new_h, image::imageops::FilterType::Triangle);

    let pad_x = (MODEL_INPUT_SIZE - new_w) as f32 / 2.0;
    let pad_y = (MODEL_INPUT_SIZE - new_h) as f32 / 2.0;
    let pad_x_i = pad_x as u32;
    let pad_y_i = pad_y as u32;

    let mut padded = GrayImage::from_pixel(
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
        image::Luma([0]),
    );
    image::imageops::overlay(&mut padded, &resized, pad_x_i as i64, pad_y_i as i64);

    // Convert to NCHW flat vec, replicate grayscale across 3 channels
    // Normalized: (pixel - 127.5) / 128.0
    let sz = MODEL_INPUT_SIZE as usize;
    let mut data = vec![0.0f32; 3 * sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let val = (padded.get_pixel(x as u32, y as u32)[0] as f32 - INPUT_MEAN) / INPUT_STD;
            data[0 * sz * sz + y * sz + x] = val;
            data[1 * sz * sz + y * sz + x] = val;
            data[2 * sz * sz + y * sz + x] = val;
        }
    }

    (data, scale, pad_x, pad_y)
}

fn nms(detections: &mut Vec<FaceDetection>) {
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(detections[i].clone());

        for j in (i + 1)..detections.len() {
            if suppressed[j] {
                continue;
            }
            if iou(&detections[i].bbox, &detections[j].bbox) > NMS_THRESHOLD {
                suppressed[j] = true;
            }
        }
    }

    *detections = keep;
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);

    inter / (area_a + area_b - inter + 1e-6)
}
