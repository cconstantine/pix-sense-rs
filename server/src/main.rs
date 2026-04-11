mod camera;
mod db;
mod face;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use image::RgbImage;
use pix_sense_common::{
    encode_frame_message, CalibrationPoint, CameraExtrinsics, DetectionConfig, FaceDetection,
    FrameMetadata, LedPoint, TrackingPoint,
};
use sqlx::{PgPool, Row as _};
use tokio::sync::{broadcast, watch};
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
    db_pool: Option<PgPool>,
    tracking_tx: broadcast::Sender<String>,
    /// Current camera extrinsics (applied to all depth detections). None = identity (camera at origin).
    extrinsics: Arc<RwLock<Option<CameraExtrinsics>>>,
    /// Calibration point pairs collected in this session (in-memory, not persisted).
    calib_points: Arc<Mutex<Vec<CalibrationPoint>>>,
    /// RealSense serial number — used as the DB key for extrinsics.
    camera_id: Arc<String>,
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

    // Query camera serial number before starting the full pipeline.
    let camera_id = Arc::new(
        camera::query_serial().unwrap_or_else(|| {
            tracing::warn!("Could not read camera serial — using 'unknown' as camera_id");
            "unknown".to_string()
        })
    );
    tracing::info!("Camera serial: {}", camera_id);

    // Connect to postgres (optional — runs without DB if DATABASE_URL is unset)
    let db_pool = db::connect().await;

    // Watch channel for broadcasting the latest encoded frame to all WebSocket clients
    let (frame_tx, frame_rx) = watch::channel(Arc::new(Vec::new()));

    // Unbounded channel for sending XYZ detections to the async DB writer task.
    // Unbounded so the sync detection thread never blocks on DB backpressure.
    let (db_tx, mut db_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<[f32; 3]>>();

    // Broadcast channel for pushing tracking_location updates to WebSocket clients.
    let (tracking_tx, _) = broadcast::channel::<String>(16);

    // Load extrinsics for this camera from DB; create and persist identity if none exist.
    let identity = CameraExtrinsics {
        r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        t: [0.0, 0.0, 0.0],
    };
    let initial_extrinsics = if let Some(pool) = &db_pool {
        if let Some(ext) = db::load_extrinsics(pool, &camera_id).await {
            tracing::info!("Loaded extrinsics for camera {}: t={:?}", camera_id, ext.t);
            ext
        } else {
            tracing::info!(
                "No extrinsics stored for camera {} — saving identity as default",
                camera_id
            );
            if let Err(e) = db::save_extrinsics(pool, &camera_id, &identity).await {
                tracing::warn!("Failed to save default extrinsics: {e:#}");
            }
            identity
        }
    } else {
        identity
    };
    let extrinsics = Arc::new(RwLock::new(Some(initial_extrinsics)));

    let calib_points: Arc<Mutex<Vec<CalibrationPoint>>> = Arc::new(Mutex::new(Vec::new()));

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    let config_clone = config.clone();
    let extrinsics_clone = extrinsics.clone();

    // Spawn camera + face detection thread (blocking, runs on a std::thread)
    let handle = std::thread::spawn(move || {
        if let Err(e) = processing_thread(frame_tx, &running_clone, config_clone, db_tx, extrinsics_clone) {
            tracing::error!("Processing thread error: {:#}", e);
        }
    });

    // Spawn async task that drains the XYZ channel and writes to postgres
    if let Some(pool) = &db_pool {
        let pool = pool.clone();
        tokio::spawn(async move {
            while let Some(xyzs) = db_rx.recv().await {
                db::write_detections(&pool, &xyzs).await;
            }
        });
    }

    // Spawn PgListener task that forwards tracking_location NOTIFYs to WebSocket clients
    if let Some(pool) = &db_pool {
        let pool = pool.clone();
        let tx = tracking_tx.clone();
        tokio::spawn(async move {
            tracking_listener(pool, tx).await;
        });
    }

    let state = AppState {
        frame_rx,
        config,
        db_pool,
        tracking_tx,
        extrinsics,
        calib_points,
        camera_id,
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/leds", get(leds_handler))
        .route("/api/calibration/camera_id", get(camera_id_handler))
        .route(
            "/api/calibration/extrinsics",
            get(get_extrinsics_handler)
                .put(put_extrinsics_handler)
                .delete(delete_extrinsics_handler),
        )
        .route(
            "/api/calibration/points",
            get(get_points_handler)
                .post(add_point_handler)
                .delete(clear_points_handler),
        )
        .route("/api/calibration/compute", post(compute_extrinsics_handler))
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
    let tracking_rx = state.tracking_tx.subscribe();
    ws.on_upgrade(|socket| handle_ws(socket, state.frame_rx, state.config, tracking_rx))
}

async fn handle_ws(
    mut socket: WebSocket,
    mut rx: watch::Receiver<Arc<Vec<u8>>>,
    config: Arc<RwLock<DetectionConfig>>,
    mut tracking_rx: broadcast::Receiver<String>,
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
            // Tracking location update from DB LISTEN — forward as JSON text
            Ok(json) = tracking_rx.recv() => {
                if socket.send(Message::Text(json.into())).await.is_err() {
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

/// Listens for PostgreSQL NOTIFY on `tracking_location_update`, queries fresh rows,
/// and broadcasts the JSON-encoded list to all connected WebSocket clients.
async fn tracking_listener(pool: PgPool, tx: broadcast::Sender<String>) {
    let Ok(database_url) = std::env::var("DATABASE_URL") else {
        tracing::warn!("DATABASE_URL not set, tracking_listener exiting");
        return;
    };
    let mut listener = match sqlx::postgres::PgListener::connect(&database_url).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("PgListener connect failed: {:#}", e);
            return;
        }
    };
    if let Err(e) = listener.listen("tracking_location_update").await {
        tracing::error!("PgListener listen failed: {:#}", e);
        return;
    }
    tracing::info!("Listening for tracking_location_update notifications");

    loop {
        match listener.recv().await {
            Ok(_notification) => {
                let rows = sqlx::query(
                    "SELECT name, x, y, z FROM tracking_locations \
                     WHERE updated_at > now() - interval '1 second'",
                )
                .fetch_all(&pool)
                .await
                .unwrap_or_default();

                let points: Vec<TrackingPoint> = rows
                    .into_iter()
                    .map(|r| TrackingPoint {
                        name: r.get("name"),
                        x: r.get("x"),
                        y: r.get("y"),
                        z: r.get("z"),
                    })
                    .collect();

                if let Ok(json) = serde_json::to_string(&points) {
                    // Ignore error — no subscribers connected is fine
                    let _ = tx.send(json);
                }
            }
            Err(e) => {
                // sqlx PgListener auto-reconnects on transient errors
                tracing::warn!("PgListener error: {:#}", e);
            }
        }
    }
}

async fn leds_handler(State(state): State<AppState>) -> impl IntoResponse {
    let Some(pool) = &state.db_pool else {
        return Json(Vec::<LedPoint>::new());
    };
    let rows = sqlx::query("SELECT x, y, z FROM leds")
        .fetch_all(pool)
        .await
        .unwrap_or_default();
    Json(
        rows.into_iter()
            .map(|r| LedPoint {
                x: r.get("x"),
                y: r.get("y"),
                z: r.get("z"),
            })
            .collect::<Vec<_>>(),
    )
}

// ---------------------------------------------------------------------------
// Calibration route handlers
// ---------------------------------------------------------------------------

/// GET /api/calibration/camera_id — returns the camera serial number as plain text.
async fn camera_id_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.camera_id.as_ref().clone()
}

/// GET /api/calibration/extrinsics — returns the current extrinsics as JSON, or 204 if none.
async fn get_extrinsics_handler(State(state): State<AppState>) -> impl IntoResponse {
    match *state.extrinsics.read().unwrap() {
        Some(ext) => Json(ext).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}

/// PUT /api/calibration/extrinsics — directly set extrinsics (e.g. from manual UI adjustment).
async fn put_extrinsics_handler(
    State(state): State<AppState>,
    Json(ext): Json<CameraExtrinsics>,
) -> impl IntoResponse {
    *state.extrinsics.write().unwrap() = Some(ext);
    if let Some(pool) = &state.db_pool {
        if let Err(e) = db::save_extrinsics(pool, &state.camera_id, &ext).await {
            tracing::warn!("Failed to persist extrinsics: {e:#}");
        }
    }
    tracing::info!("Extrinsics updated for camera {}: t={:?}", state.camera_id, ext.t);
    StatusCode::NO_CONTENT
}

/// DELETE /api/calibration/extrinsics — clears extrinsics from memory and DB.
async fn delete_extrinsics_handler(State(state): State<AppState>) -> impl IntoResponse {
    *state.extrinsics.write().unwrap() = None;
    if let Some(pool) = &state.db_pool {
        if let Err(e) = db::clear_extrinsics(pool, &state.camera_id).await {
            tracing::warn!("Failed to clear extrinsics from DB: {e:#}");
        }
    }
    tracing::info!("Extrinsics cleared for camera {}", state.camera_id);
    StatusCode::NO_CONTENT
}

/// GET /api/calibration/points — returns all accumulated calibration point pairs.
async fn get_points_handler(State(state): State<AppState>) -> impl IntoResponse {
    let pts = state.calib_points.lock().unwrap().clone();
    Json(pts)
}

/// POST /api/calibration/points — appends a calibration point pair.
async fn add_point_handler(
    State(state): State<AppState>,
    Json(pt): Json<CalibrationPoint>,
) -> impl IntoResponse {
    let mut pts = state.calib_points.lock().unwrap();
    pts.push(pt);
    let count = pts.len();
    tracing::info!(
        "Calibration point added ({} total): cam={:?} world={:?}",
        count, pt.cam, pt.world
    );
    Json(serde_json::json!({ "count": count }))
}

/// DELETE /api/calibration/points — clears all calibration point pairs.
async fn clear_points_handler(State(state): State<AppState>) -> impl IntoResponse {
    state.calib_points.lock().unwrap().clear();
    tracing::info!("Calibration points cleared");
    StatusCode::NO_CONTENT
}

/// POST /api/calibration/compute — runs Umeyama SVD on the collected point pairs,
/// saves the result to memory and DB, and returns the computed extrinsics as JSON.
/// Returns 400 if fewer than 3 point pairs are available.
async fn compute_extrinsics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let pts = state.calib_points.lock().unwrap().clone();
    if pts.len() < 3 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "Need at least 3 calibration point pairs" })),
        )
            .into_response();
    }

    match compute_rigid_transform(&pts) {
        Ok(ext) => {
            tracing::info!(
                "Computed extrinsics from {} pairs: t={:?}",
                pts.len(), ext.t
            );
            *state.extrinsics.write().unwrap() = Some(ext);
            if let Some(pool) = &state.db_pool {
                if let Err(e) = db::save_extrinsics(pool, &state.camera_id, &ext).await {
                    tracing::warn!("Failed to persist extrinsics: {e:#}");
                }
            }
            Json(ext).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Umeyama rigid body registration (SVD)
// ---------------------------------------------------------------------------

/// Compute the optimal rigid body transform R, t such that p_world ≈ R * p_cam + t.
///
/// Uses the Kabsch/Umeyama SVD method. Requires ≥3 non-collinear point pairs.
/// The det-correction step ensures R is a proper rotation (det = +1).
fn compute_rigid_transform(pts: &[CalibrationPoint]) -> Result<CameraExtrinsics> {
    use nalgebra::{Matrix3, SVD, Vector3};

    let n = pts.len() as f64;

    // Compute centroids
    let mut cam_c = Vector3::zeros();
    let mut world_c = Vector3::zeros();
    for p in pts {
        cam_c += Vector3::new(p.cam[0] as f64, p.cam[1] as f64, p.cam[2] as f64);
        world_c += Vector3::new(p.world[0] as f64, p.world[1] as f64, p.world[2] as f64);
    }
    cam_c /= n;
    world_c /= n;

    // Cross-covariance matrix H = Σ (p_cam - c_cam) * (p_world - c_world)^T
    let mut h = Matrix3::<f64>::zeros();
    for p in pts {
        let pc = Vector3::new(p.cam[0] as f64, p.cam[1] as f64, p.cam[2] as f64) - cam_c;
        let pw = Vector3::new(p.world[0] as f64, p.world[1] as f64, p.world[2] as f64) - world_c;
        h += pc * pw.transpose();
    }

    let svd = SVD::new(h, true, true);
    let u = svd.u.ok_or_else(|| anyhow::anyhow!("SVD failed to converge (U)"))?;
    let v_t = svd.v_t.ok_or_else(|| anyhow::anyhow!("SVD failed to converge (V)"))?;

    // Ensure proper rotation: if det(V U^T) < 0, negate the last singular vector
    let mut d = Matrix3::<f64>::identity();
    if (v_t.transpose() * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }

    // R = V * D * U^T
    let r_mat = v_t.transpose() * d * u.transpose();

    // t = c_world - R * c_cam
    let t_vec = world_c - r_mat * cam_c;

    let r = [
        [r_mat[(0, 0)] as f32, r_mat[(0, 1)] as f32, r_mat[(0, 2)] as f32],
        [r_mat[(1, 0)] as f32, r_mat[(1, 1)] as f32, r_mat[(1, 2)] as f32],
        [r_mat[(2, 0)] as f32, r_mat[(2, 1)] as f32, r_mat[(2, 2)] as f32],
    ];
    let t = [t_vec[0] as f32, t_vec[1] as f32, t_vec[2] as f32];

    Ok(CameraExtrinsics { r, t })
}

fn processing_thread(
    tx: watch::Sender<Arc<Vec<u8>>>,
    running: &AtomicBool,
    config: Arc<RwLock<DetectionConfig>>,
    db_tx: tokio::sync::mpsc::UnboundedSender<Vec<[f32; 3]>>,
    extrinsics: Arc<RwLock<Option<CameraExtrinsics>>>,
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
        let rgb_faces: Vec<_> = rgb_faces_raw
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

        let ir_faces: Vec<_> = ir_faces_raw
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

        // Send XYZ coordinates to the DB writer task (non-blocking).
        // Apply camera extrinsics to convert from camera frame to world frame.
        let xyzs: Vec<[f32; 3]> = {
            let ext = extrinsics.read().unwrap();
            rgb_faces
                .iter()
                .chain(ir_faces.iter())
                .filter_map(|f| f.xyz)
                .map(|p| ext.as_ref().map_or(p, |e| e.apply(p)))
                .collect()
        };
        if !xyzs.is_empty() {
            let _ = db_tx.send(xyzs);
        }

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
