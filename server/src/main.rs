mod camera;
mod face;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
use pix_sense_common::{encode_frame_message, FaceDetection, FrameMetadata};
use tokio::sync::watch;
use tower_http::services::ServeDir;

use camera::Camera;
use face::FaceDetector;

const MODEL_PATH: &str = "models/scrfd_10g_bnkps.onnx";

/// Data produced by the processing thread
struct FrameData {
    rgb: RgbImage,
    ir: image::GrayImage,
    depth: image::GrayImage,
    depth_size: [u32; 2],
    rgb_faces: Vec<FaceDetection>,
    ir_faces: Vec<FaceDetection>,
}

#[derive(Clone)]
struct AppState {
    frame_rx: watch::Receiver<Arc<Vec<u8>>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Verify model exists
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("ERROR: Model not found at '{}'", MODEL_PATH);
        eprintln!("Run ./setup.sh to download the SCRFD model.");
        std::process::exit(1);
    }

    // Watch channel for broadcasting the latest encoded frame to all WebSocket clients
    let (frame_tx, frame_rx) = watch::channel(Arc::new(Vec::new()));
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Spawn camera + face detection thread (blocking, runs on a std::thread)
    let handle = std::thread::spawn(move || {
        if let Err(e) = processing_thread(frame_tx, &running_clone) {
            tracing::error!("Processing thread error: {:#}", e);
        }
    });

    let state = AppState { frame_rx };

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
    ws.on_upgrade(|socket| handle_ws(socket, state.frame_rx))
}

async fn handle_ws(mut socket: WebSocket, mut rx: watch::Receiver<Arc<Vec<u8>>>) {
    loop {
        // Wait for a new frame
        if rx.changed().await.is_err() {
            break; // sender dropped
        }

        let frame = rx.borrow_and_update().clone();
        if frame.is_empty() {
            continue; // no frame yet
        }

        if socket
            .send(Message::Binary((*frame).clone().into()))
            .await
            .is_err()
        {
            break; // client disconnected
        }
    }
    tracing::info!("WebSocket client disconnected");
}

fn processing_thread(
    tx: watch::Sender<Arc<Vec<u8>>>,
    running: &AtomicBool,
) -> Result<()> {
    // Load model first — CUDA graph compilation can take a while,
    // and the RealSense pipeline may stall if frames aren't consumed.
    tracing::info!("Loading face detection model...");
    let mut detector = FaceDetector::new(MODEL_PATH)?;
    tracing::info!("Face detector ready");

    tracing::info!("Initializing camera...");
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

        let t0 = std::time::Instant::now();
        let rgb_faces = match detector.detect_rgb(&frames.rgb) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("RGB face detection error: {:#}", e);
                Vec::new()
            }
        };
        let rgb_detect_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = std::time::Instant::now();
        let ir_faces = match detector.detect_gray(&frames.ir) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("IR face detection error: {:#}", e);
                Vec::new()
            }
        };
        let ir_detect_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let data = FrameData {
            rgb: frames.rgb,
            ir: frames.ir,
            depth: frames.depth,
            depth_size: frames.depth_size,
            rgb_faces,
            ir_faces,
        };

        if frame_tx.send(data).is_err() {
            break; // encode thread exited
        }

        let detect_ms = t_loop.elapsed().as_secs_f64() * 1000.0;
        frame_count += 1;

        // Log timing every 2 seconds
        if log_interval.elapsed().as_secs_f64() >= 2.0 {
            tracing::info!(
                "frame {}: capture={:.1}ms  rgb={:.1}ms  ir={:.1}ms  detect_total={:.1}ms ({:.1} fps)",
                frame_count,
                capture_ms,
                rgb_detect_ms,
                ir_detect_ms,
                detect_ms,
                1000.0 / detect_ms,
            );
            log_interval = std::time::Instant::now();
        }
    }

    encode_running.store(false, Ordering::Relaxed);
    drop(frame_tx);
    let _ = encode_handle.join();

    tracing::info!("Processing thread shutting down");
    Ok(())
}

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
    };

    encode_frame_message(&rgb_jpeg, &ir_jpeg, &depth_jpeg, &metadata)
}
