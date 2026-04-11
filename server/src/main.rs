mod camera;
mod face;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use image::RgbImage;
use pix_sense_common::{
    encode_frame_message, DetectionConfig, FaceDetection, FrameMetadata,
};
use tokio::sync::watch;
use tower_http::services::ServeDir;

use camera::{Camera, CameraIntrinsics};
use face::{FaceDetector, HeadDetector};

const SCRFD_MODEL_PATH: &str = "models/scrfd_10g_bnkps.onnx";
const YOLO_HEAD_MODEL_PATH: &str = "models/yolov8n_head.onnx";

/// Data produced by the processing thread
struct FrameData {
    rgb: RgbImage,
    ir: image::GrayImage,
    depth: image::GrayImage,
    depth_size: [u32; 2],
    rgb_faces: Vec<FaceDetection>,
    ir_faces: Vec<FaceDetection>,
    active_config: DetectionConfig,
}

#[derive(Clone)]
struct AppState {
    frame_rx: watch::Receiver<Arc<Vec<u8>>>,
    config: Arc<RwLock<DetectionConfig>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Verify models exist
    for path in [YOLO_HEAD_MODEL_PATH, SCRFD_MODEL_PATH] {
        if !std::path::Path::new(path).exists() {
            eprintln!("ERROR: Model not found at '{}'", path);
            eprintln!("Run ./setup.sh to download models.");
            std::process::exit(1);
        }
    }

    let config = Arc::new(RwLock::new(DetectionConfig::default()));

    // Watch channel for broadcasting the latest encoded frame to all WebSocket clients
    let (frame_tx, frame_rx) = watch::channel(Arc::new(Vec::new()));
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    let config_clone = config.clone();

    // Spawn camera + face detection thread (blocking, runs on a std::thread)
    let handle = std::thread::spawn(move || {
        if let Err(e) = processing_thread(frame_tx, &running_clone, config_clone) {
            tracing::error!("Processing thread error: {:#}", e);
        }
    });

    let state = AppState { frame_rx, config };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new("client/dist"))
        .with_state(state);

    let bind_addr = "0.0.0.0:3000";
    tracing::info!("Server listening on {}", bind_addr);
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app).await?;

    // Signal processing thread to stop
    running.store(false, Ordering::Relaxed);
    let _ = handle.join();

    Ok(())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    tracing::info!("New WebSocket connection");
    ws.on_upgrade(|socket| handle_ws(socket, state.frame_rx, state.config))
}

async fn handle_ws(
    mut socket: WebSocket,
    mut rx: watch::Receiver<Arc<Vec<u8>>>,
    config: Arc<RwLock<DetectionConfig>>,
) {
    loop {
        tokio::select! {
            // New frame available — forward to client
            changed = rx.changed() => {
                if changed.is_err() {
                    break; // sender dropped
                }
                let frame = rx.borrow_and_update().clone();
                if frame.is_empty() {
                    continue;
                }
                if socket.send(Message::Binary((*frame).clone().into())).await.is_err() {
                    break; // client disconnected
                }
            }
            // Message from client — handle config updates
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(cfg) = serde_json::from_str::<DetectionConfig>(&text) {
                            *config.write().unwrap() = cfg;
                            tracing::info!(
                                "Config updated: algo={:?} stream={:?}",
                                cfg.algo, cfg.stream
                            );
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
        }
    }
    tracing::info!("WebSocket client disconnected");
}

fn processing_thread(
    tx: watch::Sender<Arc<Vec<u8>>>,
    running: &AtomicBool,
    config: Arc<RwLock<DetectionConfig>>,
) -> Result<()> {
    // Load models first — CUDA graph compilation can take a while,
    // and the RealSense pipeline may stall if frames aren't consumed.
    tracing::info!("Loading YOLO head detection model...");
    let mut head_detector = HeadDetector::new(YOLO_HEAD_MODEL_PATH)?;
    tracing::info!("YOLO head detector ready");

    tracing::info!("Loading SCRFD face landmark model...");
    let mut face_detector = FaceDetector::new(SCRFD_MODEL_PATH)?;
    tracing::info!("SCRFD face detector ready");

    tracing::info!("Initializing camera (with RGB→IR alignment)...");
    let mut camera = Camera::new()?;
    tracing::info!("Camera initialized");

    // Pipeline: detection thread sends FrameData to encode thread via bounded channel.
    // Capacity 1 so we never buffer stale frames — detection blocks until encode consumes.
    let (frame_tx, frame_rx) = std::sync::mpsc::sync_channel::<FrameData>(1);

    // Encode thread: JPEG-encodes frames (CPU/NEON) while detection uses GPU
    let encode_running = Arc::new(AtomicBool::new(true));
    let encode_running_clone = encode_running.clone();
    let encode_handle = std::thread::spawn(move || {
        encode_thread(frame_rx, tx, &encode_running_clone);
    });

    let mut frame_count = 0u64;
    let mut log_interval = std::time::Instant::now();

    while running.load(Ordering::Relaxed) {
        let t_loop = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let frames = match camera.capture() {
            Ok(Some(f)) => f,
            Ok(None) => continue,
            Err(e) => {
                tracing::warn!("Camera capture error: {:#}", e);
                continue;
            }
        };
        let capture_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let cfg = *config.read().unwrap();

        let (rgb_faces_raw, ir_faces_raw, timing) = run_detection(
            &mut head_detector,
            &mut face_detector,
            &frames,
            cfg,
        );

        // Attach XYZ coordinates to every detection
        let rgb_faces = rgb_faces_raw
            .into_iter()
            .map(|mut f| {
                f.xyz = compute_xyz(
                    &f.bbox,
                    &frames.depth_raw,
                    frames.depth_size[0],
                    frames.depth_size[1],
                    &frames.intrinsics,
                );
                f
            })
            .collect();

        let ir_faces = ir_faces_raw
            .into_iter()
            .map(|mut f| {
                f.xyz = compute_xyz(
                    &f.bbox,
                    &frames.depth_raw,
                    frames.depth_size[0],
                    frames.depth_size[1],
                    &frames.intrinsics,
                );
                f
            })
            .collect();

        let data = FrameData {
            rgb: frames.rgb,
            ir: frames.ir,
            depth: frames.depth,
            depth_size: frames.depth_size,
            rgb_faces,
            ir_faces,
            active_config: cfg,
        };

        if frame_tx.send(data).is_err() {
            break; // encode thread exited
        }

        let detect_ms = t_loop.elapsed().as_secs_f64() * 1000.0;
        frame_count += 1;

        // Log timing every 2 seconds
        if log_interval.elapsed().as_secs_f64() >= 2.0 {
            tracing::info!(
                "frame {} total={:.1}ms ({:.1} fps)  capture={:.1}ms  algo={:?}  stream={:?}",
                frame_count, detect_ms, 1000.0 / detect_ms, capture_ms,
                cfg.algo, cfg.stream,
            );
            if let Some(t) = &timing {
                tracing::info!(
                    "  RGB: pre={:.1}ms  trt={:.1}ms  post={:.1}ms  heads={}  scrfd={:.1}ms  total={:.1}ms",
                    t.rgb_pre_ms, t.rgb_infer_ms, t.rgb_post_ms,
                    t.rgb_heads, t.rgb_scrfd_ms, t.rgb_total_ms,
                );
                tracing::info!(
                    "  IR:  pre={:.1}ms  trt={:.1}ms  post={:.1}ms  heads={}  scrfd={:.1}ms  total={:.1}ms",
                    t.ir_pre_ms, t.ir_infer_ms, t.ir_post_ms,
                    t.ir_heads, t.ir_scrfd_ms, t.ir_total_ms,
                );
            }
            log_interval = std::time::Instant::now();
        }
    }

    encode_running.store(false, Ordering::Relaxed);
    drop(frame_tx);
    let _ = encode_handle.join();

    tracing::info!("Processing thread shutting down");
    Ok(())
}

// ---------------------------------------------------------------------------
// Timing breakdown
// ---------------------------------------------------------------------------

struct DetectionTiming {
    rgb_pre_ms: f64,
    rgb_infer_ms: f64,
    rgb_post_ms: f64,
    rgb_heads: usize,
    rgb_scrfd_ms: f64,
    rgb_total_ms: f64,
    ir_pre_ms: f64,
    ir_infer_ms: f64,
    ir_post_ms: f64,
    ir_heads: usize,
    ir_scrfd_ms: f64,
    ir_total_ms: f64,
}

// ---------------------------------------------------------------------------
// Detection dispatcher
// ---------------------------------------------------------------------------

use pix_sense_common::{DetectionAlgo, StreamSelection};

/// Run detection on the appropriate stream(s) according to `cfg`.
/// Returns (rgb_faces, ir_faces, optional_timing).
fn run_detection(
    head: &mut HeadDetector,
    face: &mut FaceDetector,
    frames: &camera::CameraFrames,
    cfg: DetectionConfig,
) -> (Vec<FaceDetection>, Vec<FaceDetection>, Option<DetectionTiming>) {
    let run_rgb = matches!(cfg.stream, StreamSelection::Rgb | StreamSelection::Both);
    let run_ir = matches!(cfg.stream, StreamSelection::Ir | StreamSelection::Both);

    let (rgb_faces, rgb_t) = if run_rgb {
        detect_stream_rgb(head, face, &frames.rgb, cfg.algo)
    } else {
        (Vec::new(), None)
    };

    let (ir_faces, ir_t) = if run_ir {
        detect_stream_ir(head, face, &frames.ir, cfg.algo)
    } else {
        (Vec::new(), None)
    };

    // Build combined timing if both streams ran
    let timing = match (rgb_t, ir_t) {
        (Some(r), Some(i)) => Some(DetectionTiming {
            rgb_pre_ms: r.0, rgb_infer_ms: r.1, rgb_post_ms: r.2,
            rgb_heads: r.3, rgb_scrfd_ms: r.4, rgb_total_ms: r.5,
            ir_pre_ms: i.0, ir_infer_ms: i.1, ir_post_ms: i.2,
            ir_heads: i.3, ir_scrfd_ms: i.4, ir_total_ms: i.5,
        }),
        (Some(r), None) => Some(DetectionTiming {
            rgb_pre_ms: r.0, rgb_infer_ms: r.1, rgb_post_ms: r.2,
            rgb_heads: r.3, rgb_scrfd_ms: r.4, rgb_total_ms: r.5,
            ir_pre_ms: 0.0, ir_infer_ms: 0.0, ir_post_ms: 0.0,
            ir_heads: 0, ir_scrfd_ms: 0.0, ir_total_ms: 0.0,
        }),
        (None, Some(i)) => Some(DetectionTiming {
            rgb_pre_ms: 0.0, rgb_infer_ms: 0.0, rgb_post_ms: 0.0,
            rgb_heads: 0, rgb_scrfd_ms: 0.0, rgb_total_ms: 0.0,
            ir_pre_ms: i.0, ir_infer_ms: i.1, ir_post_ms: i.2,
            ir_heads: i.3, ir_scrfd_ms: i.4, ir_total_ms: i.5,
        }),
        (None, None) => None,
    };

    (rgb_faces, ir_faces, timing)
}

/// (pre_ms, infer_ms, post_ms, num_heads, scrfd_ms, total_ms)
type StreamTiming = (f64, f64, f64, usize, f64, f64);

fn detect_stream_rgb(
    head: &mut HeadDetector,
    face: &mut FaceDetector,
    image: &RgbImage,
    algo: DetectionAlgo,
) -> (Vec<FaceDetection>, Option<StreamTiming>) {
    let t_total = std::time::Instant::now();

    match algo {
        DetectionAlgo::YoloHead => {
            let t0 = std::time::Instant::now();
            let (heads, yt) = match head.detect_rgb(image) {
                Ok(r) => r,
                Err(e) => { tracing::warn!("YOLO RGB error: {:#}", e); return (Vec::new(), None); }
            };
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let pre = elapsed - yt.trt_infer_ms - yt.postprocess_ms;
            let total = t_total.elapsed().as_secs_f64() * 1000.0;
            (heads.clone(), Some((pre, yt.trt_infer_ms, yt.postprocess_ms, heads.len(), 0.0, total)))
        }
        DetectionAlgo::ScrfdFace => {
            let t0 = std::time::Instant::now();
            let (faces, ft) = match face.detect_rgb(image) {
                Ok(r) => r,
                Err(e) => { tracing::warn!("SCRFD RGB error: {:#}", e); return (Vec::new(), None); }
            };
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let pre = elapsed - ft.trt_infer_ms - ft.postprocess_ms;
            let total = t_total.elapsed().as_secs_f64() * 1000.0;
            (faces.clone(), Some((pre, ft.trt_infer_ms, ft.postprocess_ms, faces.len(), 0.0, total)))
        }
        DetectionAlgo::YoloHeadScrfdLandmarks => {
            let t0 = std::time::Instant::now();
            let (mut heads, yt) = match head.detect_rgb(image) {
                Ok(r) => r,
                Err(e) => { tracing::warn!("YOLO RGB error: {:#}", e); return (Vec::new(), None); }
            };
            let yolo_elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let pre = yolo_elapsed - yt.trt_infer_ms - yt.postprocess_ms;

            let t_scrfd = std::time::Instant::now();
            for h in &mut heads {
                let crop = crop_rgb(image, &h.bbox);
                if let Ok(Some(lms)) = face.detect_rgb_crop(&crop) {
                    // Translate landmarks from crop-local to full-image coordinates
                    let ox = h.bbox[0];
                    let oy = h.bbox[1];
                    h.landmarks = Some(lms.map(|lm| pix_sense_common::FaceLandmark {
                        x: lm.x + ox,
                        y: lm.y + oy,
                    }));
                }
            }
            let scrfd_ms = t_scrfd.elapsed().as_secs_f64() * 1000.0;
            let total = t_total.elapsed().as_secs_f64() * 1000.0;
            let n = heads.len();
            (heads, Some((pre, yt.trt_infer_ms, yt.postprocess_ms, n, scrfd_ms, total)))
        }
    }
}

fn detect_stream_ir(
    head: &mut HeadDetector,
    face: &mut FaceDetector,
    image: &image::GrayImage,
    algo: DetectionAlgo,
) -> (Vec<FaceDetection>, Option<StreamTiming>) {
    let t_total = std::time::Instant::now();

    match algo {
        DetectionAlgo::YoloHead => {
            let t0 = std::time::Instant::now();
            let (heads, yt) = match head.detect_gray(image) {
                Ok(r) => r,
                Err(e) => { tracing::warn!("YOLO IR error: {:#}", e); return (Vec::new(), None); }
            };
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let pre = elapsed - yt.trt_infer_ms - yt.postprocess_ms;
            let total = t_total.elapsed().as_secs_f64() * 1000.0;
            (heads.clone(), Some((pre, yt.trt_infer_ms, yt.postprocess_ms, heads.len(), 0.0, total)))
        }
        DetectionAlgo::ScrfdFace => {
            let t0 = std::time::Instant::now();
            let (faces, ft) = match face.detect_gray(image) {
                Ok(r) => r,
                Err(e) => { tracing::warn!("SCRFD IR error: {:#}", e); return (Vec::new(), None); }
            };
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let pre = elapsed - ft.trt_infer_ms - ft.postprocess_ms;
            let total = t_total.elapsed().as_secs_f64() * 1000.0;
            (faces.clone(), Some((pre, ft.trt_infer_ms, ft.postprocess_ms, faces.len(), 0.0, total)))
        }
        DetectionAlgo::YoloHeadScrfdLandmarks => {
            let t0 = std::time::Instant::now();
            let (mut heads, yt) = match head.detect_gray(image) {
                Ok(r) => r,
                Err(e) => { tracing::warn!("YOLO IR error: {:#}", e); return (Vec::new(), None); }
            };
            let yolo_elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let pre = yolo_elapsed - yt.trt_infer_ms - yt.postprocess_ms;

            let t_scrfd = std::time::Instant::now();
            for h in &mut heads {
                let crop = crop_gray(image, &h.bbox);
                if let Ok(Some(lms)) = face.detect_gray_crop(&crop) {
                    let ox = h.bbox[0];
                    let oy = h.bbox[1];
                    h.landmarks = Some(lms.map(|lm| pix_sense_common::FaceLandmark {
                        x: lm.x + ox,
                        y: lm.y + oy,
                    }));
                }
            }
            let scrfd_ms = t_scrfd.elapsed().as_secs_f64() * 1000.0;
            let total = t_total.elapsed().as_secs_f64() * 1000.0;
            let n = heads.len();
            (heads, Some((pre, yt.trt_infer_ms, yt.postprocess_ms, n, scrfd_ms, total)))
        }
    }
}

// ---------------------------------------------------------------------------
// XYZ deprojection
// ---------------------------------------------------------------------------

/// Sample a 5×5 window of depth values around the face bounding-box centre,
/// take the median of valid readings, and deproject to camera-frame XYZ (metres).
fn compute_xyz(
    bbox: &[f32; 4],
    depth_raw: &[u16],
    depth_w: u32,
    depth_h: u32,
    intr: &CameraIntrinsics,
) -> Option<[f32; 3]> {
    if depth_raw.is_empty() {
        return None;
    }

    let cx_px = (bbox[0] + bbox[2]) / 2.0;
    let cy_px = (bbox[1] + bbox[3]) / 2.0;

    let mut samples: Vec<u16> = Vec::with_capacity(25);
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let x = (cx_px as i32 + dx).clamp(0, depth_w as i32 - 1) as u32;
            let y = (cy_px as i32 + dy).clamp(0, depth_h as i32 - 1) as u32;
            let d = depth_raw[(y * depth_w + x) as usize];
            // Valid range: 300–5000 mm (matches depth_to_gray)
            if d >= 300 && d <= 5000 {
                samples.push(d);
            }
        }
    }

    if samples.is_empty() {
        return None;
    }

    samples.sort_unstable();
    let z_mm = samples[samples.len() / 2] as f32;
    let z = z_mm / 1000.0; // metres

    let x = (cx_px - intr.ppx) * z / intr.fx;
    let y = (cy_px - intr.ppy) * z / intr.fy;

    Some([x, y, z])
}

// ---------------------------------------------------------------------------
// Image crop helpers
// ---------------------------------------------------------------------------

/// Crop an RGB image to the given bounding box [x1, y1, x2, y2].
fn crop_rgb(image: &RgbImage, bbox: &[f32; 4]) -> RgbImage {
    let x1 = (bbox[0] as u32).min(image.width().saturating_sub(1));
    let y1 = (bbox[1] as u32).min(image.height().saturating_sub(1));
    let x2 = (bbox[2] as u32).min(image.width());
    let y2 = (bbox[3] as u32).min(image.height());
    let w = (x2 - x1).max(1);
    let h = (y2 - y1).max(1);
    image::imageops::crop_imm(image, x1, y1, w, h).to_image()
}

/// Crop a grayscale image to the given bounding box [x1, y1, x2, y2].
fn crop_gray(image: &image::GrayImage, bbox: &[f32; 4]) -> image::GrayImage {
    let x1 = (bbox[0] as u32).min(image.width().saturating_sub(1));
    let y1 = (bbox[1] as u32).min(image.height().saturating_sub(1));
    let x2 = (bbox[2] as u32).min(image.width());
    let y2 = (bbox[3] as u32).min(image.height());
    let w = (x2 - x1).max(1);
    let h = (y2 - y1).max(1);
    image::imageops::crop_imm(image, x1, y1, w, h).to_image()
}

// ---------------------------------------------------------------------------
// Encode thread
// ---------------------------------------------------------------------------

fn encode_thread(
    rx: std::sync::mpsc::Receiver<FrameData>,
    tx: watch::Sender<Arc<Vec<u8>>>,
    running: &AtomicBool,
) {
    let mut compressor = match turbojpeg::Compressor::new() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to create JPEG compressor: {}", e);
            return;
        }
    };
    compressor.set_quality(75).ok();

    while running.load(Ordering::Relaxed) {
        let data = match rx.recv() {
            Ok(d) => d,
            Err(_) => break, // sender dropped
        };

        let msg = encode_frame(&mut compressor, &data);
        let _ = tx.send(Arc::new(msg));
    }
}

fn encode_frame(compressor: &mut turbojpeg::Compressor, data: &FrameData) -> Vec<u8> {
    compressor.set_subsamp(turbojpeg::Subsamp::Sub2x2).ok();
    let rgb_jpeg = compressor
        .compress_to_vec(turbojpeg::Image {
            pixels: data.rgb.as_raw().as_slice(),
            width: data.rgb.width() as usize,
            pitch: data.rgb.width() as usize * 3,
            height: data.rgb.height() as usize,
            format: turbojpeg::PixelFormat::RGB,
        })
        .expect("JPEG encode failed");

    compressor.set_subsamp(turbojpeg::Subsamp::Gray).ok();
    let ir_jpeg = compressor
        .compress_to_vec(turbojpeg::Image {
            pixels: data.ir.as_raw().as_slice(),
            width: data.ir.width() as usize,
            pitch: data.ir.width() as usize,
            height: data.ir.height() as usize,
            format: turbojpeg::PixelFormat::GRAY,
        })
        .expect("JPEG encode failed");

    let depth_jpeg = compressor
        .compress_to_vec(turbojpeg::Image {
            pixels: data.depth.as_raw().as_slice(),
            width: data.depth.width() as usize,
            pitch: data.depth.width() as usize,
            height: data.depth.height() as usize,
            format: turbojpeg::PixelFormat::GRAY,
        })
        .expect("JPEG encode failed");

    let metadata = FrameMetadata {
        rgb_faces: data.rgb_faces.clone(),
        ir_faces: data.ir_faces.clone(),
        rgb_size: [data.rgb.width(), data.rgb.height()],
        ir_size: [data.ir.width(), data.ir.height()],
        depth_size: data.depth_size,
        active_config: data.active_config,
    };

    encode_frame_message(&rgb_jpeg, &ir_jpeg, &depth_jpeg, &metadata)
}
