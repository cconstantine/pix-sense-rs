use anyhow::{Context, Result};
use image::RgbImage;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

const MODEL_INPUT_SIZE: u32 = 640;
const CONF_THRESHOLD: f32 = 0.25;
const IOU_THRESHOLD: f32 = 0.45;
pub const KPT_CONF_THRESHOLD: f32 = 0.5;

/// COCO 17 keypoints
pub const SKELETON: [(usize, usize); 16] = [
    (0, 1),   // nose -> left_eye
    (0, 2),   // nose -> right_eye
    (1, 3),   // left_eye -> left_ear
    (2, 4),   // right_eye -> right_ear
    (5, 6),   // left_shoulder -> right_shoulder
    (5, 7),   // left_shoulder -> left_elbow
    (7, 9),   // left_elbow -> left_wrist
    (6, 8),   // right_shoulder -> right_elbow
    (8, 10),  // right_elbow -> right_wrist
    (5, 11),  // left_shoulder -> left_hip
    (6, 12),  // right_shoulder -> right_hip
    (11, 12), // left_hip -> right_hip
    (11, 13), // left_hip -> left_knee
    (13, 15), // left_knee -> left_ankle
    (12, 14), // right_hip -> right_knee
    (14, 16), // right_knee -> right_ankle
];

#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct Pose {
    pub bbox: [f32; 4], // x1, y1, x2, y2
    pub confidence: f32,
    pub keypoints: Vec<Keypoint>,
}

pub struct PoseEstimator {
    session: Session,
}

impl PoseEstimator {
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

        tracing::info!("Pose estimation model loaded from {}", model_path);

        Ok(PoseEstimator { session })
    }

    /// Run pose estimation on an RGB image.
    pub fn estimate(&mut self, image: &RgbImage) -> Result<Vec<Pose>> {
        let (img_w, img_h) = (image.width(), image.height());

        let (input_data, scale, pad_x, pad_y) = preprocess(image);

        let input_tensor = Tensor::from_array((
            [1usize, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize],
            input_data.into_boxed_slice(),
        ))
        .context("Failed to create input tensor")?;

        let outputs = self
            .session
            .run(ort::inputs![input_tensor])
            .context("Inference failed")?;

        let (shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor")?;

        let num_detections = shape[2] as usize;

        let mut poses = Vec::new();

        for det in 0..num_detections {
            let cx = output_data[0 * num_detections + det];
            let cy = output_data[1 * num_detections + det];
            let w = output_data[2 * num_detections + det];
            let h = output_data[3 * num_detections + det];
            let confidence = output_data[4 * num_detections + det];

            if confidence < CONF_THRESHOLD {
                continue;
            }

            let x1 = ((cx - w / 2.0 - pad_x) / scale).clamp(0.0, img_w as f32);
            let y1 = ((cy - h / 2.0 - pad_y) / scale).clamp(0.0, img_h as f32);
            let x2 = ((cx + w / 2.0 - pad_x) / scale).clamp(0.0, img_w as f32);
            let y2 = ((cy + h / 2.0 - pad_y) / scale).clamp(0.0, img_h as f32);

            let mut keypoints = Vec::with_capacity(17);
            for k in 0..17 {
                let kx_idx = (5 + k * 3) * num_detections + det;
                let ky_idx = (5 + k * 3 + 1) * num_detections + det;
                let kc_idx = (5 + k * 3 + 2) * num_detections + det;

                let kx = ((output_data[kx_idx] - pad_x) / scale).clamp(0.0, img_w as f32);
                let ky = ((output_data[ky_idx] - pad_y) / scale).clamp(0.0, img_h as f32);
                let kc = output_data[kc_idx];

                keypoints.push(Keypoint {
                    x: kx,
                    y: ky,
                    confidence: kc,
                });
            }

            poses.push(Pose {
                bbox: [x1, y1, x2, y2],
                confidence,
                keypoints,
            });
        }

        nms(&mut poses);

        Ok(poses)
    }
}

fn preprocess(image: &RgbImage) -> (Vec<f32>, f32, f32, f32) {
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
        image::Rgb([114, 114, 114]),
    );
    image::imageops::overlay(&mut padded, &resized, pad_x_i as i64, pad_y_i as i64);

    let sz = MODEL_INPUT_SIZE as usize;
    let mut data = vec![0.0f32; 3 * sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let pixel = padded.get_pixel(x as u32, y as u32);
            data[0 * sz * sz + y * sz + x] = pixel[0] as f32 / 255.0;
            data[1 * sz * sz + y * sz + x] = pixel[1] as f32 / 255.0;
            data[2 * sz * sz + y * sz + x] = pixel[2] as f32 / 255.0;
        }
    }

    (data, scale, pad_x, pad_y)
}

fn nms(poses: &mut Vec<Pose>) {
    poses.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; poses.len()];

    for i in 0..poses.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(poses[i].clone());

        for j in (i + 1)..poses.len() {
            if suppressed[j] {
                continue;
            }
            if iou(&poses[i].bbox, &poses[j].bbox) > IOU_THRESHOLD {
                suppressed[j] = true;
            }
        }
    }

    *poses = keep;
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
