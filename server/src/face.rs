use anyhow::{Context, Result, bail};
use image::{GrayImage, RgbImage};
use pix_sense_common::{FaceDetection, FaceLandmark};
use std::ffi::{CStr, CString, c_char};
use std::os::raw::c_int;
use std::path::Path;

const MODEL_INPUT_SIZE: u32 = 640;
const CONF_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.4;
const INPUT_MEAN: f32 = 127.5;
const INPUT_STD: f32 = 128.0;

const STRIDES: [u32; 3] = [8, 16, 32];
const NUM_ANCHORS: u32 = 2;

// FFI bindings to trt_wrapper.cpp
#[repr(C)]
struct TrtEngine {
    _private: [u8; 0],
}

extern "C" {
    fn trt_engine_create(
        onnx_path: *const c_char,
        cache_path: *const c_char,
        fp16: c_int,
    ) -> *mut TrtEngine;
    fn trt_engine_destroy(engine: *mut TrtEngine);
    fn trt_engine_num_io(engine: *mut TrtEngine) -> i32;
    fn trt_engine_tensor_name(engine: *mut TrtEngine, index: i32) -> *const c_char;
    fn trt_engine_tensor_is_input(engine: *mut TrtEngine, name: *const c_char) -> i32;
    fn trt_engine_tensor_shape(
        engine: *mut TrtEngine,
        name: *const c_char,
        dims: *mut i64,
        nb_dims: *mut i32,
    );
    fn trt_engine_infer(
        engine: *mut TrtEngine,
        input_data: *const f32,
        input_bytes: usize,
        output_ptrs: *mut *mut f32,
        output_sizes: *mut usize,
        num_outputs: i32,
    ) -> i32;
}

/// Maps output tensor indices by type and stride order.
/// Indices refer to positions in the output-only tensor list.
struct OutputMap {
    /// Score tensor indices, ordered by stride [8, 16, 32]
    scores: [usize; 3],
    /// Bbox tensor indices, ordered by stride [8, 16, 32]
    bboxes: [usize; 3],
    /// Keypoint tensor indices (if present), ordered by stride [8, 16, 32]
    kps: Option<[usize; 3]>,
}

pub struct FaceDetector {
    engine: *mut TrtEngine,
    output_map: OutputMap,
    num_outputs: usize,
    output_sizes: Vec<usize>,
    // Pre-allocated buffers reused across inference calls
    output_bufs: Vec<Vec<f32>>,
}

unsafe impl Send for FaceDetector {}

impl Drop for FaceDetector {
    fn drop(&mut self) {
        unsafe {
            trt_engine_destroy(self.engine);
        }
    }
}

impl FaceDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        // Derive cache path from model path: models/foo.onnx -> models/foo.engine
        let cache_path = Path::new(model_path).with_extension("engine");
        let onnx_cstr =
            CString::new(model_path).context("Invalid model path")?;
        let cache_cstr =
            CString::new(cache_path.to_str().unwrap_or("")).context("Invalid cache path")?;

        let engine = unsafe { trt_engine_create(onnx_cstr.as_ptr(), cache_cstr.as_ptr(), 1) };
        if engine.is_null() {
            bail!("Failed to create TensorRT engine from {}", model_path);
        }

        // Query I/O tensors and build output map
        let num_io = unsafe { trt_engine_num_io(engine) };
        let mut outputs: Vec<(usize, String, Vec<i64>)> = Vec::new(); // (output_index, name, shape)
        let mut out_idx = 0usize;

        for i in 0..num_io {
            let name_ptr = unsafe { trt_engine_tensor_name(engine, i) };
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_str()
                .unwrap_or("")
                .to_string();
            let name_c = CString::new(name.as_str()).unwrap();
            let is_input = unsafe { trt_engine_tensor_is_input(engine, name_c.as_ptr()) };
            if is_input == 1 {
                continue;
            }

            let mut dims = [0i64; 8];
            let mut nb_dims = 0i32;
            unsafe {
                trt_engine_tensor_shape(engine, name_c.as_ptr(), dims.as_mut_ptr(), &mut nb_dims);
            }
            let shape: Vec<i64> = dims[..nb_dims as usize].to_vec();
            outputs.push((out_idx, name, shape));
            out_idx += 1;
        }

        tracing::info!("TensorRT engine has {} output tensors", outputs.len());
        for (idx, name, shape) in &outputs {
            tracing::info!("  output[{}] '{}' shape={:?}", idx, name, shape);
        }

        // Classify outputs by last dimension: 1=score, 4=bbox, 10=kps
        let mut score_outputs: Vec<(usize, i64)> = Vec::new(); // (out_idx, num_anchors)
        let mut bbox_outputs: Vec<(usize, i64)> = Vec::new();
        let mut kps_outputs: Vec<(usize, i64)> = Vec::new();

        for (idx, _name, shape) in &outputs {
            let last_dim = *shape.last().unwrap_or(&0);
            // For SCRFD, shapes are [1, N, 1], [1, N, 4], [1, N, 10]
            // or sometimes [1, N] for scores
            let n = if shape.len() >= 2 { shape[shape.len() - 2] } else { shape[0] };
            match last_dim {
                1 => score_outputs.push((*idx, n)),
                4 => bbox_outputs.push((*idx, n)),
                10 => kps_outputs.push((*idx, n)),
                _ => {
                    // If last dim matches total anchors for a stride, it might be a flat score tensor
                    // Try treating it as score if shape is [1, N]
                    if shape.len() == 2 {
                        score_outputs.push((*idx, last_dim));
                    } else {
                        tracing::warn!(
                            "Unknown output tensor shape {:?} at index {}",
                            shape,
                            idx
                        );
                    }
                }
            }
        }

        // Sort each group by descending N (stride 8 has the most anchors)
        score_outputs.sort_by(|a, b| b.1.cmp(&a.1));
        bbox_outputs.sort_by(|a, b| b.1.cmp(&a.1));
        kps_outputs.sort_by(|a, b| b.1.cmp(&a.1));

        if score_outputs.len() < 3 || bbox_outputs.len() < 3 {
            bail!(
                "Expected at least 3 score and 3 bbox outputs, got {} scores and {} bboxes",
                score_outputs.len(),
                bbox_outputs.len()
            );
        }

        let output_map = OutputMap {
            scores: [score_outputs[0].0, score_outputs[1].0, score_outputs[2].0],
            bboxes: [bbox_outputs[0].0, bbox_outputs[1].0, bbox_outputs[2].0],
            kps: if kps_outputs.len() >= 3 {
                Some([kps_outputs[0].0, kps_outputs[1].0, kps_outputs[2].0])
            } else {
                None
            },
        };

        // Compute output buffer sizes
        let num_outputs = outputs.len();
        let mut output_sizes = Vec::with_capacity(num_outputs);
        for (_, _, shape) in &outputs {
            let vol: i64 = shape.iter().product();
            output_sizes.push(vol as usize * std::mem::size_of::<f32>());
        }

        tracing::info!(
            "Face detection model loaded via TensorRT (FP16 enabled, {} outputs)",
            num_outputs
        );

        let output_bufs = output_sizes
            .iter()
            .map(|&sz| vec![0.0f32; sz / std::mem::size_of::<f32>()])
            .collect();

        Ok(FaceDetector {
            engine,
            output_map,
            num_outputs,
            output_sizes,
            output_bufs,
        })
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
        let mut output_ptrs: Vec<*mut f32> = self
            .output_bufs
            .iter_mut()
            .map(|buf| buf.as_mut_ptr())
            .collect();

        let mut output_sizes_bytes: Vec<usize> = self.output_sizes.clone();

        let input_bytes = input_data.len() * std::mem::size_of::<f32>();
        let ret = unsafe {
            trt_engine_infer(
                self.engine,
                input_data.as_ptr(),
                input_bytes,
                output_ptrs.as_mut_ptr(),
                output_sizes_bytes.as_mut_ptr(),
                self.num_outputs as i32,
            )
        };
        if ret != 0 {
            bail!("TensorRT inference failed with error code {}", ret);
        }

        // Decode detections
        let mut detections = Vec::new();

        for (stride_idx, &stride) in STRIDES.iter().enumerate() {
            let feat_h = MODEL_INPUT_SIZE / stride;
            let feat_w = MODEL_INPUT_SIZE / stride;

            let scores_data = &self.output_bufs[self.output_map.scores[stride_idx]];
            let bbox_data = &self.output_bufs[self.output_map.bboxes[stride_idx]];
            let kps_data = self.output_map.kps.map(|kps| &self.output_bufs[kps[stride_idx]]);

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

                    let bbox_offset = anchor_idx * 4;
                    let d0 = bbox_data[bbox_offset] * stride as f32;
                    let d1 = bbox_data[bbox_offset + 1] * stride as f32;
                    let d2 = bbox_data[bbox_offset + 2] * stride as f32;
                    let d3 = bbox_data[bbox_offset + 3] * stride as f32;

                    let x1 = ((anchor_x - d0 - pad_x) / scale).clamp(0.0, img_w as f32);
                    let y1 = ((anchor_y - d1 - pad_y) / scale).clamp(0.0, img_h as f32);
                    let x2 = ((anchor_x + d2 - pad_x) / scale).clamp(0.0, img_w as f32);
                    let y2 = ((anchor_y + d3 - pad_y) / scale).clamp(0.0, img_h as f32);

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
            data[y * sz + x] = (pixel[0] as f32 - INPUT_MEAN) / INPUT_STD;
            data[sz * sz + y * sz + x] = (pixel[1] as f32 - INPUT_MEAN) / INPUT_STD;
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

    let sz = MODEL_INPUT_SIZE as usize;
    let mut data = vec![0.0f32; 3 * sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            let val = (padded.get_pixel(x as u32, y as u32)[0] as f32 - INPUT_MEAN) / INPUT_STD;
            data[y * sz + x] = val;
            data[sz * sz + y * sz + x] = val;
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
