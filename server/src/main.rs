mod camera;
mod db;
mod face;
mod person_selector;
mod renderer;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Path, State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use image::RgbImage;
use pix_sense_common::{
    encode_frame_message, CalibrationPoint, CameraExtrinsics, ClientMessage, DetectionConfig,
    FaceDetection, FrameMetadata, LedPoint, Pattern, PatternUpdate, ServerMessage, TrackingPoint,
};
use sqlx::{PgPool, Row as _};
use tokio::sync::{broadcast, watch};
use tower_http::services::ServeDir;

use prometheus::{
    register_gauge_vec, register_histogram, register_histogram_vec, register_int_counter,
    Encoder, GaugeVec, Histogram, HistogramVec, IntCounter, TextEncoder,
};

use camera::{Camera, CameraIntrinsics};
use face::{FaceDetector, HeadDetector};

const SCRFD_MODEL_PATH: &str = "models/scrfd_10g_bnkps.onnx";
const YOLO_HEAD_MODEL_PATH: &str = "models/yolov8n_head.onnx";

// ---------------------------------------------------------------------------
// Prometheus metrics
// ---------------------------------------------------------------------------

const TIMING_BUCKETS: &[f64] = &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0];

struct Metrics {
    frame_count: IntCounter,
    frame_time_ms: Histogram,
    capture_time_ms: Histogram,
    pipeline_ms: HistogramVec,
    heads_detected: GaugeVec,
}

impl Metrics {
    fn new() -> Self {
        let opts = |name: &str, help: &str| {
            prometheus::HistogramOpts::new(name, help).buckets(TIMING_BUCKETS.to_vec())
        };
        Self {
            frame_count: register_int_counter!(
                "pix_frame_count_total",
                "Total number of frames processed"
            )
            .unwrap(),
            frame_time_ms: register_histogram!(opts(
                "pix_frame_time_ms",
                "Total frame processing time in milliseconds"
            ))
            .unwrap(),
            capture_time_ms: register_histogram!(opts(
                "pix_capture_time_ms",
                "Camera capture time in milliseconds"
            ))
            .unwrap(),
            pipeline_ms: register_histogram_vec!(
                opts(
                    "pix_pipeline_stage_ms",
                    "Pipeline stage duration in milliseconds"
                ),
                &["stream", "stage"]
            )
            .unwrap(),
            heads_detected: register_gauge_vec!(
                "pix_heads_detected",
                "Number of heads detected per frame",
                &["stream"]
            )
            .unwrap(),
        }
    }
}

/// Data produced by the processing thread
struct FrameData {
    rgb: RgbImage,
    ir: image::GrayImage,
    rgb_faces: Vec<FaceDetection>,
    ir_faces: Vec<FaceDetection>,
    active_config: DetectionConfig,
    roi_rect: [u32; 4],
    tracked_rgb_idx: Option<usize>,
    tracked_ir_idx: Option<usize>,
}

// ---------------------------------------------------------------------------
// ROI tracker — keeps the detection crop window centered on the last face
// ---------------------------------------------------------------------------

struct RoiTracker {
    /// Smoothed crop center in full-image pixel coordinates.
    /// `None` means scanning mode (no face tracked).
    center: Option<(f32, f32)>,
    /// Consecutive frames with zero detections.
    frames_without_detection: u32,
    /// Coast on last known position for this many misses before scanning.
    max_coast_frames: u32,
    /// Current index into the scan pattern (cycles through positions).
    scan_index: usize,
    /// EMA smoothing factor (higher = more responsive).
    ema_alpha: f32,
}

/// Horizontal scan positions (center-x values) for a 1280-wide frame with a
/// 640-wide crop.  Three positions tile the frame with 50 % overlap so every
/// pixel is covered within 3 frames.
const SCAN_POSITIONS_X: [f32; 3] = [320.0, 640.0, 960.0];

impl RoiTracker {
    fn new() -> Self {
        Self {
            center: None,
            frames_without_detection: 0,
            max_coast_frames: 30,
            scan_index: 0,
            ema_alpha: 0.3,
        }
    }

    /// Return the crop rectangle `[x1, y1, x2, y2]` clamped to frame bounds.
    fn crop_rect(&self, frame_w: u32, frame_h: u32, crop_size: u32) -> [u32; 4] {
        let (cx, cy) = match self.center {
            Some(c) => c,
            None => {
                let sx = SCAN_POSITIONS_X[self.scan_index % SCAN_POSITIONS_X.len()];
                (sx, frame_h as f32 / 2.0)
            }
        };

        let half = crop_size as f32 / 2.0;
        // Clamp so the full crop fits inside the frame when possible.
        let max_x1 = (frame_w as f32 - crop_size as f32).max(0.0);
        let max_y1 = (frame_h as f32 - crop_size as f32).max(0.0);
        let x1 = (cx - half).max(0.0).min(max_x1) as u32;
        let y1 = (cy - half).max(0.0).min(max_y1) as u32;
        let x2 = (x1 + crop_size).min(frame_w);
        let y2 = (y1 + crop_size).min(frame_h);

        [x1, y1, x2, y2]
    }

    /// Feed back this frame's detections (already in full-image coordinates).
    fn update(&mut self, detections: &[FaceDetection]) {
        if detections.is_empty() {
            self.frames_without_detection += 1;
            if self.frames_without_detection >= self.max_coast_frames || self.center.is_none() {
                // Switch to / stay in scanning mode.
                self.center = None;
                self.scan_index += 1;
            }
            return;
        }

        self.frames_without_detection = 0;

        // Track the highest-confidence detection.
        let best = detections
            .iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .unwrap();
        let new_cx = (best.bbox[0] + best.bbox[2]) / 2.0;
        let new_cy = (best.bbox[1] + best.bbox[3]) / 2.0;

        match self.center {
            Some((old_cx, old_cy)) => {
                let dist = ((new_cx - old_cx).powi(2) + (new_cy - old_cy).powi(2)).sqrt();
                if dist > 100.0 {
                    self.center = Some((new_cx, new_cy));
                } else {
                    let a = self.ema_alpha;
                    self.center = Some((
                        old_cx + a * (new_cx - old_cx),
                        old_cy + a * (new_cy - old_cy),
                    ));
                }
            }
            None => {
                self.center = Some((new_cx, new_cy));
            }
        }
    }
}

#[derive(Clone)]
struct AppState {
    frame_rx: watch::Receiver<Arc<Vec<u8>>>,
    config: Arc<RwLock<DetectionConfig>>,
    db_pool: Option<PgPool>,
    tracking_tx: broadcast::Sender<String>,
    /// Latest tracked person world position forwarded to the GL renderer.
    #[allow(dead_code)] // kept in AppState to own the sender; all reads go through tracking_listener
    tracking_pos_tx: watch::Sender<Option<[f32; 3]>>,
    /// Latest LED color frame from the renderer, pre-serialised as JSON for WebSocket broadcast.
    led_colors_rx: watch::Receiver<Option<String>>,
    /// Current camera extrinsics (applied to all depth detections). None = identity (camera at origin).
    extrinsics: Arc<RwLock<Option<CameraExtrinsics>>>,
    /// Pending manual person-selection target (world frame). Drained by the processing
    /// thread once per frame to override `PersonSelector`'s automatic pick.
    pending_selection: Arc<Mutex<Option<[f32; 3]>>>,
    /// Calibration point pairs collected in this session (in-memory, not persisted).
    calib_points: Arc<Mutex<Vec<CalibrationPoint>>>,
    /// RealSense serial number — used as the DB key for extrinsics.
    camera_id: Arc<String>,
    /// Sculpture name from SCULPTURE_NAME env var — used for pattern activation.
    sculpture_name: Arc<String>,
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

    // Load persisted detection config from DB, falling back to defaults if unavailable.
    let initial_config = if let Some(pool) = &db_pool {
        db::load_detection_config(pool).await.unwrap_or_default()
    } else {
        DetectionConfig::default()
    };
    tracing::info!(
        "Detection config: algo={:?} stream={:?}",
        initial_config.algo,
        initial_config.stream
    );
    let config = Arc::new(RwLock::new(initial_config));

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

    let pending_selection: Arc<Mutex<Option<[f32; 3]>>> = Arc::new(Mutex::new(None));

    let metrics = Arc::new(Metrics::new());

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    let config_clone = config.clone();
    let extrinsics_clone = extrinsics.clone();
    let pending_selection_clone = pending_selection.clone();
    let metrics_clone = metrics.clone();

    // Spawn camera + face detection thread (blocking, runs on a std::thread)
    let handle = std::thread::spawn(move || {
        if let Err(e) = processing_thread(
            frame_tx,
            &running_clone,
            config_clone,
            db_tx,
            extrinsics_clone,
            pending_selection_clone,
            metrics_clone,
        ) {
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

    // Watch channel for broadcasting the latest tracked person position to the GL renderer.
    let (tracking_pos_tx, tracking_pos_rx) = watch::channel::<Option<[f32; 3]>>(None);

    // Watch channel for broadcasting LED colors from the renderer to WebSocket clients.
    // Value is pre-serialised JSON so the WS handler can forward it directly.
    let (led_colors_tx, led_colors_rx) = watch::channel::<Option<String>>(None);

    // Spawn PgListener task that forwards tracking_location NOTIFYs to WebSocket clients
    // and updates the tracking position watch channel for the renderer.
    if let Some(pool) = &db_pool {
        let pool = pool.clone();
        let ws_tx = tracking_tx.clone();
        let pos_tx = tracking_pos_tx.clone();
        tokio::spawn(async move {
            tracking_listener(pool, ws_tx, pos_tx).await;
        });
    }

    let sculpture_name = Arc::new(
        std::env::var("SCULPTURE_NAME").unwrap_or_else(|_| "default".to_string()),
    );
    tracing::info!("Sculpture name: {}", sculpture_name);

    // Spawn the headless GLSL renderer thread (requires DB to be configured).
    if let Some(pool) = &db_pool {
        let pool = pool.clone();
        let rx = tracking_pos_rx;
        let sname = (*sculpture_name).clone();
        let running_r = running.clone();
        std::thread::Builder::new()
            .name("gl-renderer".into())
            .spawn(move || renderer::run(pool, rx, led_colors_tx, running_r, sname))
            .expect("spawn gl-renderer thread");
    }

    let state = AppState {
        frame_rx,
        config,
        db_pool,
        tracking_tx,
        tracking_pos_tx,
        led_colors_rx,
        extrinsics,
        pending_selection,
        calib_points,
        camera_id,
        sculpture_name,
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
        // Pattern management
        .route(
            "/api/patterns",
            get(list_patterns_handler).post(create_pattern_handler),
        )
        .route("/api/patterns/active", get(get_active_pattern_handler))
        .route(
            "/api/patterns/{name}",
            get(get_pattern_handler)
                .put(update_pattern_handler)
                .delete(delete_pattern_handler),
        )
        .route("/api/patterns/{name}/activate", post(activate_pattern_handler))
        .route("/metrics", get(metrics_handler))
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
    ws.on_upgrade(|socket| {
        handle_ws(
            socket,
            state.frame_rx,
            state.config,
            tracking_rx,
            state.led_colors_rx,
            state.db_pool,
            state.extrinsics,
            state.pending_selection,
        )
    })
}

async fn handle_ws(
    mut socket: WebSocket,
    mut rx: watch::Receiver<Arc<Vec<u8>>>,
    config: Arc<RwLock<DetectionConfig>>,
    mut tracking_rx: broadcast::Receiver<String>,
    mut led_colors_rx: watch::Receiver<Option<String>>,
    db_pool: Option<PgPool>,
    extrinsics: Arc<RwLock<Option<CameraExtrinsics>>>,
    pending_selection: Arc<Mutex<Option<[f32; 3]>>>,
) {
    // Send the current detection config immediately so the client UI reflects
    // the persisted setting rather than its hard-coded default.
    // Copy out of the lock guard before awaiting (guard is !Send across await).
    let current_config = *config.read().unwrap();
    if let Ok(json) = serde_json::to_string(&ServerMessage::Config(current_config)) {
        if socket.send(Message::Text(json.into())).await.is_err() {
            return;
        }
    }

    // Once the renderer's sender is dropped, stop polling that arm rather than
    // disconnecting the client (which would break frame/tracking delivery too).
    let mut led_colors_closed = false;

    loop {
        tokio::select! {
            // New camera frame available — forward as binary
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
            // LED color update from renderer — forward as JSON text.
            // Guard disabled once the renderer sender is gone so this arm doesn't
            // spin-loop on a permanently-closed channel and starve the others.
            changed = led_colors_rx.changed(), if !led_colors_closed => {
                if changed.is_err() {
                    led_colors_closed = true; // renderer stopped; skip this arm hereafter
                } else {
                    // Clone out of the guard before awaiting (guard is not Send).
                    let json = led_colors_rx.borrow_and_update().clone();
                    if let Some(json) = json {
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                }
            }
            // Message from client — handle config updates and manual selections
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<ClientMessage>(&text) {
                            Ok(ClientMessage::Config(cfg)) => {
                                *config.write().unwrap() = cfg;
                                tracing::info!(
                                    "Config updated: algo={:?} stream={:?}",
                                    cfg.algo, cfg.stream
                                );
                                if let Some(pool) = db_pool.clone() {
                                    tokio::spawn(async move {
                                        db::save_detection_config(&pool, cfg).await;
                                    });
                                }
                            }
                            Ok(ClientMessage::SelectPerson { xyz }) => {
                                let world = extrinsics
                                    .read()
                                    .unwrap()
                                    .as_ref()
                                    .map_or(xyz, |e| e.apply(xyz));
                                *pending_selection.lock().unwrap() = Some(world);
                                tracing::info!(
                                    "Manual person selection: cam={:?} world={:?}",
                                    xyz, world
                                );
                            }
                            Err(e) => {
                                tracing::warn!("Unrecognised client message: {e}");
                            }
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
/// broadcasts the JSON-encoded list to all connected WebSocket clients, and updates
/// the renderer's tracking position watch channel.
async fn tracking_listener(
    pool: PgPool,
    ws_tx: broadcast::Sender<String>,
    pos_tx: watch::Sender<Option<[f32; 3]>>,
) {
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

                // Update renderer with the first tracked position (primary person).
                let renderer_pos = points.first().map(|p| [p.x, p.y, p.z]);
                let _ = pos_tx.send(renderer_pos);

                if let Ok(json) =
                    serde_json::to_string(&ServerMessage::Tracking(points))
                {
                    // Ignore error — no subscribers connected is fine
                    let _ = ws_tx.send(json);
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
    let rows = sqlx::query(
        "SELECT l.x, l.y, l.z FROM leds l \
         JOIN fadecandies f ON f.id = l.fadecandy_id \
         ORDER BY f.id, l.idx",
    )
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

// ---------------------------------------------------------------------------
// Pattern route handlers
// ---------------------------------------------------------------------------

async fn list_patterns_handler(State(state): State<AppState>) -> impl IntoResponse {
    let Some(pool) = &state.db_pool else {
        return Json(Vec::<Pattern>::new()).into_response();
    };
    Json(db::list_patterns(pool).await).into_response()
}

async fn create_pattern_handler(
    State(state): State<AppState>,
    Json(pattern): Json<Pattern>,
) -> impl IntoResponse {
    let Some(pool) = &state.db_pool else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    match db::create_pattern(pool, &pattern).await {
        Ok(()) => StatusCode::CREATED.into_response(),
        Err(e)
            if e.as_database_error()
                .map(|d| d.is_unique_violation())
                .unwrap_or(false) =>
        {
            (
                StatusCode::CONFLICT,
                Json(serde_json::json!({ "error": "Pattern name already exists" })),
            )
                .into_response()
        }
        Err(e) => {
            tracing::warn!("create_pattern error: {e:#}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn get_pattern_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let Some(pool) = &state.db_pool else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    match db::get_pattern(pool, &name).await {
        Some(p) => Json(p).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn update_pattern_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(update): Json<PatternUpdate>,
) -> impl IntoResponse {
    let Some(pool) = &state.db_pool else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    match db::update_pattern(pool, &name, &update).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            tracing::warn!("update_pattern error: {e:#}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn delete_pattern_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let Some(pool) = &state.db_pool else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    match db::delete_pattern(pool, &name).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            tracing::warn!("delete_pattern error: {e:#}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn activate_pattern_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let Some(pool) = &state.db_pool else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    match db::set_active_pattern(pool, &state.sculpture_name, &name).await {
        Ok(()) => {
            tracing::info!("Active pattern set to '{}' for sculpture '{}'", name, state.sculpture_name);
            StatusCode::NO_CONTENT.into_response()
        }
        Err(e) => {
            tracing::warn!("set_active_pattern error: {e:#}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}

async fn get_active_pattern_handler(State(state): State<AppState>) -> impl IntoResponse {
    let Some(pool) = &state.db_pool else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    match db::get_active_pattern(pool, &state.sculpture_name).await {
        Some(name) => name.into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    }
}

async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let families = prometheus::gather();
    let mut buf = Vec::new();
    encoder.encode(&families, &mut buf).unwrap();
    (
        [(
            axum::http::header::CONTENT_TYPE,
            encoder.format_type().to_owned(),
        )],
        buf,
    )
}

fn processing_thread(
    tx: watch::Sender<Arc<Vec<u8>>>,
    running: &AtomicBool,
    config: Arc<RwLock<DetectionConfig>>,
    db_tx: tokio::sync::mpsc::UnboundedSender<Vec<[f32; 3]>>,
    extrinsics: Arc<RwLock<Option<CameraExtrinsics>>>,
    pending_selection: Arc<Mutex<Option<[f32; 3]>>>,
    metrics: Arc<Metrics>,
) -> Result<()> {
    // Load models first — CUDA graph compilation can take a while,
    // and the RealSense pipeline may stall if frames aren't consumed.
    tracing::info!("Loading YOLO head detection model...");
    let mut head_detector = HeadDetector::new(YOLO_HEAD_MODEL_PATH)?;
    tracing::info!("YOLO head detector ready");

    tracing::info!("Loading SCRFD face landmark model...");
    let mut face_detector = FaceDetector::new(SCRFD_MODEL_PATH)?;
    tracing::info!("SCRFD face detector ready");

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

    let mut roi_tracker = RoiTracker::new();
    let mut person_selector = person_selector::PersonSelector::new();

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

        let roi_rect = roi_tracker.crop_rect(frames.rgb.width(), frames.rgb.height(), 640);

        let (rgb_faces_raw, ir_faces_raw, timing) = run_detection(
            &mut head_detector,
            &mut face_detector,
            &frames,
            cfg,
            roi_rect,
        );

        // Update ROI tracker with the active stream's detections.
        match cfg.stream {
            StreamSelection::Rgb => roi_tracker.update(&rgb_faces_raw),
            StreamSelection::Ir => roi_tracker.update(&ir_faces_raw),
        }

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

        // Build indexed world-frame positions with source tracking so we can
        // identify which face in which stream the PersonSelector picks.
        let face_refs: Vec<(bool, usize, [f32; 3])> = {
            let ext = extrinsics.read().unwrap();
            let mut refs = Vec::new();
            for (i, f) in rgb_faces.iter().enumerate() {
                if let Some(p) = f.xyz {
                    refs.push((true, i, ext.as_ref().map_or(p, |e| e.apply(p))));
                }
            }
            for (i, f) in ir_faces.iter().enumerate() {
                if let Some(p) = f.xyz {
                    refs.push((false, i, ext.as_ref().map_or(p, |e| e.apply(p))));
                }
            }
            refs
        };
        let world_xyzs: Vec<[f32; 3]> = face_refs.iter().map(|&(_, _, xyz)| xyz).collect();

        // Apply any pending manual selection from the UI before the selector runs.
        if let Some(target) = pending_selection.lock().unwrap().take() {
            person_selector.lock_to(target);
        }

        let (tracked_rgb_idx, tracked_ir_idx) =
            if let Some((sel_idx, selected)) = person_selector.select(&world_xyzs) {
                let _ = db_tx.send(vec![selected]);
                let (is_rgb, face_idx, _) = face_refs[sel_idx];
                if is_rgb {
                    (Some(face_idx), None)
                } else {
                    (None, Some(face_idx))
                }
            } else {
                (None, None)
            };

        let data = FrameData {
            rgb: frames.rgb,
            ir: frames.ir,
            rgb_faces,
            ir_faces,
            active_config: cfg,
            roi_rect,
            tracked_rgb_idx,
            tracked_ir_idx,
        };

        if frame_tx.send(data).is_err() {
            break; // encode thread exited
        }

        let detect_ms = t_loop.elapsed().as_secs_f64() * 1000.0;

        // Update Prometheus metrics
        metrics.frame_count.inc();
        metrics.frame_time_ms.observe(detect_ms);
        metrics.capture_time_ms.observe(capture_ms);

        if let Some(t) = &timing {
            metrics.pipeline_ms.with_label_values(&["rgb", "pre"]).observe(t.rgb_pre_ms);
            metrics.pipeline_ms.with_label_values(&["rgb", "infer"]).observe(t.rgb_infer_ms);
            metrics.pipeline_ms.with_label_values(&["rgb", "post"]).observe(t.rgb_post_ms);
            metrics.pipeline_ms.with_label_values(&["rgb", "scrfd"]).observe(t.rgb_scrfd_ms);
            metrics.pipeline_ms.with_label_values(&["rgb", "total"]).observe(t.rgb_total_ms);
            metrics.heads_detected.with_label_values(&["rgb"]).set(t.rgb_heads as f64);

            metrics.pipeline_ms.with_label_values(&["ir", "pre"]).observe(t.ir_pre_ms);
            metrics.pipeline_ms.with_label_values(&["ir", "infer"]).observe(t.ir_infer_ms);
            metrics.pipeline_ms.with_label_values(&["ir", "post"]).observe(t.ir_post_ms);
            metrics.pipeline_ms.with_label_values(&["ir", "scrfd"]).observe(t.ir_scrfd_ms);
            metrics.pipeline_ms.with_label_values(&["ir", "total"]).observe(t.ir_total_ms);
            metrics.heads_detected.with_label_values(&["ir"]).set(t.ir_heads as f64);
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
    roi_rect: [u32; 4],
) -> (Vec<FaceDetection>, Vec<FaceDetection>, Option<DetectionTiming>) {
    let run_rgb = matches!(cfg.stream, StreamSelection::Rgb);
    let run_ir = matches!(cfg.stream, StreamSelection::Ir);

    let offset_x = roi_rect[0] as f32;
    let offset_y = roi_rect[1] as f32;

    let (mut rgb_faces, rgb_t) = if run_rgb {
        let crop = roi_crop_rgb(&frames.rgb, &roi_rect);
        detect_stream_rgb(head, face, &crop, cfg.algo)
    } else {
        (Vec::new(), None)
    };
    translate_detections(&mut rgb_faces, offset_x, offset_y);

    let (mut ir_faces, ir_t) = if run_ir {
        let crop = roi_crop_gray(&frames.ir, &roi_rect);
        detect_stream_ir(head, face, &crop, cfg.algo)
    } else {
        (Vec::new(), None)
    };
    translate_detections(&mut ir_faces, offset_x, offset_y);

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

/// Sample an adaptive window of depth values around the face bounding-box centre,
/// take the median of valid readings, and deproject to camera-frame XYZ (metres).
/// The window scales to ~30% of the face bbox (clamped to 5–21 px per axis) so that
/// distant (small) faces still get adequate depth coverage.
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

    // Adaptive sampling: ~30 % of bbox, clamped to [5, 21] px per axis
    let face_w = (bbox[2] - bbox[0]).max(1.0);
    let face_h = (bbox[3] - bbox[1]).max(1.0);
    let sample_w = ((face_w * 0.3) as i32).clamp(5, 21);
    let sample_h = ((face_h * 0.3) as i32).clamp(5, 21);
    let half_w = sample_w / 2;
    let half_h = sample_h / 2;

    let mut samples: Vec<u16> = Vec::with_capacity((sample_w * sample_h) as usize);
    for dy in -half_h..=half_h {
        for dx in -half_w..=half_w {
            let x = (cx_px as i32 + dx).clamp(0, depth_w as i32 - 1) as u32;
            let y = (cy_px as i32 + dy).clamp(0, depth_h as i32 - 1) as u32;
            let d = depth_raw[(y * depth_w + x) as usize];
            // Valid range: 300–9000 mm (D455 rated to ~6 m usable, ~14 m max)
            if d >= 300 && d <= 9000 {
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

    let x = -((cx_px - intr.ppx) * z / intr.fx);
    let y = -((cy_px - intr.ppy) * z / intr.fy);

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

/// Crop an RGB image to a ROI rectangle [x1, y1, x2, y2] (u32 pixel coords).
fn roi_crop_rgb(image: &RgbImage, rect: &[u32; 4]) -> RgbImage {
    let [x1, y1, x2, y2] = *rect;
    image::imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image()
}

/// Crop a grayscale image to a ROI rectangle [x1, y1, x2, y2] (u32 pixel coords).
fn roi_crop_gray(image: &image::GrayImage, rect: &[u32; 4]) -> image::GrayImage {
    let [x1, y1, x2, y2] = *rect;
    image::imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image()
}

/// Shift detection bboxes and landmarks from crop-local to full-image coordinates.
fn translate_detections(detections: &mut [FaceDetection], offset_x: f32, offset_y: f32) {
    for det in detections.iter_mut() {
        det.bbox[0] += offset_x;
        det.bbox[1] += offset_y;
        det.bbox[2] += offset_x;
        det.bbox[3] += offset_y;
        if let Some(ref mut lms) = det.landmarks {
            for lm in lms.iter_mut() {
                lm.x += offset_x;
                lm.y += offset_y;
            }
        }
    }
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

    let metadata = FrameMetadata {
        rgb_faces: data.rgb_faces.clone(),
        ir_faces: data.ir_faces.clone(),
        rgb_size: [data.rgb.width(), data.rgb.height()],
        ir_size: [data.ir.width(), data.ir.height()],
        active_config: data.active_config,
        roi_rect: Some(data.roi_rect),
        tracked_rgb_idx: data.tracked_rgb_idx,
        tracked_ir_idx: data.tracked_ir_idx,
    };

    encode_frame_message(&rgb_jpeg, &ir_jpeg, &metadata)
}
