mod overlay;
mod scene3d;

use eframe::egui;
use ewebsock::{WsEvent, WsMessage, WsReceiver, WsSender};
use pix_sense_common::{
    decode_frame_message, CalibrationPoint, CameraExtrinsics, ClientMessage, DetectionAlgo,
    DetectionConfig, FaceDetection, LedPoint, Pattern, PatternUpdate, SculptureSettings,
    ServerMessage, StreamSelection, TrackingPoint,
};
use scene3d::SceneRenderer;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let web_options = eframe::WebOptions::default();

    // Get the canvas element by ID
    let document = web_sys::window()
        .expect("no window")
        .document()
        .expect("no document");
    let canvas = document
        .get_element_by_id("the_canvas_id")
        .expect("no canvas element")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("element is not a canvas");

    wasm_bindgen_futures::spawn_local(async move {
        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(App::new(cc)))),
            )
            .await
            .expect("Failed to start eframe");
    });

    Ok(())
}

fn get_ws_url() -> String {
    let window = web_sys::window().expect("no window");
    let location = window.location();
    let protocol = location.protocol().unwrap_or_else(|_| "http:".into());
    let host = location.host().unwrap_or_else(|_| "localhost:3000".into());

    let ws_protocol = if protocol == "https:" { "wss:" } else { "ws:" };
    format!("{}//{}/ws", ws_protocol, host)
}


const DEFAULT_PATTERN_GLSL: &str = "\
void main() {\n\
    vec2 uv = gl_FragCoord.xy / resolution;\n\
    fragColor = vec4(uv, 0.5 + 0.5 * sin(time * 0.001), 1.0);\n\
}\n";

/// What drives the 3D scene's orbit camera.
#[derive(Clone, PartialEq)]
enum SceneLock {
    /// User-controlled orbit (drag = rotate, scroll = zoom).
    Free,
    /// Camera orbits around the named tracked person's world position.
    Tracked(String),
}

/// Where the LED renderer's tracking location comes from. Orthogonal to
/// `SceneLock` — the scene camera and the tracking source are independent.
#[derive(Copy, Clone, PartialEq, Eq)]
enum TrackingSource {
    /// Server pushes real person detections from the depth camera (default).
    DepthCamera,
    /// Server uses the scene camera's orbit eye position as the tracking
    /// location. The client streams `VirtualLocation` updates each frame.
    SceneCamera,
    /// Server stops pushing new tracking positions to the renderer, so the
    /// LEDs hold whatever state was last sent. Useful for inspecting the
    /// sculpture from other angles without the LEDs updating.
    Frozen,
}

/// Which capture workflow is active inside the calibration side panel.
#[derive(Copy, Clone, PartialEq, Eq)]
enum CalibrationMode {
    /// Operator types the world-frame XYZ and clicks a tracking dot in the 3D
    /// scene to capture its camera-frame XYZ.
    PickPoint,
    /// Operator stands at a known physical spot, drags translation sliders until
    /// their tracked marker aligns with the spot, then snapshots the currently-
    /// tracked person's (cam, world) pair in one click. Rotation falls out of
    /// the SVD solve once ≥3 pairs are collected.
    StandInView,
}

struct PatternDraft {
    original_name: String,
    name: String,
    glsl: String,
    enabled: bool,
    overscan: bool,
    is_new: bool,
}

fn is_valid_pattern_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

struct App {
    ws_sender: WsSender,
    ws_receiver: WsReceiver,
    rgb_texture: Option<egui::TextureHandle>,
    ir_texture: Option<egui::TextureHandle>,
    latest_rgb_faces: Vec<FaceDetection>,
    latest_ir_faces: Vec<FaceDetection>,
    rgb_size: [u32; 2],
    ir_size: [u32; 2],
    roi_rect: Option<[u32; 4]>,
    tracked_rgb_idx: Option<usize>,
    tracked_ir_idx: Option<usize>,
    fps_counter: FpsCounter,
    connected: bool,
    /// Config the user has selected in the UI (sent to server on change).
    local_config: DetectionConfig,
    /// Config the server reports as active (echoed back in FrameMetadata).
    active_config: DetectionConfig,
    // 3D scene data
    leds: Vec<LedPoint>,
    tracking: Vec<TrackingPoint>,
    /// Current LED colors from the renderer, indexed parallel to `leds`.
    led_colors: Vec<[u8; 3]>,
    /// Populated once by the /api/leds fetch; None until the response arrives.
    led_pending: Rc<RefCell<Option<Vec<LedPoint>>>>,
    // 3D orbit camera state
    scene_yaw: f32,
    scene_pitch: f32,
    scene_zoom: f32,
    // Window visibility
    show_cameras: bool,
    show_scene: bool,
    show_patterns: bool,
    /// Which viewpoint the 3D scene's orbit camera is driven by.
    scene_lock: SceneLock,
    /// Where the LED renderer's tracking location comes from.
    tracking_source: TrackingSource,
    /// Throttle timestamp for `send_virtual_location` (egui `ctx.input.time`).
    /// Used while `tracking_source == SceneCamera` to rate-limit WS updates.
    last_virtual_send: f64,
    // Calibration state
    calib_mode: bool,
    /// Which capture workflow is active within calibration mode.
    calib_sub_mode: CalibrationMode,
    /// Point pairs collected so far (mirrors server-side list for this session).
    calib_points: Vec<CalibrationPoint>,
    /// World-frame XYZ typed by the user (pending pairing with a cam point).
    calib_world_input: [String; 3],
    /// Camera-frame point captured by clicking a tracking dot.
    calib_pending_cam: Option<[f32; 3]>,
    /// Last error message from a stand-in-view capture attempt (shown briefly).
    calib_capture_error: Rc<RefCell<Option<String>>>,
    /// Current extrinsics (None = identity / uncalibrated).
    extrinsics: Option<CameraExtrinsics>,
    /// Camera serial number fetched from server.
    camera_id: String,
    // Async fetch pending cells (same pattern as led_pending)
    extrinsics_pending: Rc<RefCell<Option<Option<CameraExtrinsics>>>>,
    points_pending: Rc<RefCell<Option<Vec<CalibrationPoint>>>>,
    camera_id_pending: Rc<RefCell<Option<String>>>,
    // Manual extrinsics editor state
    /// Translation in metres, synced to/from `extrinsics.t`.
    manual_t: [f32; 3],
    /// Euler XYZ rotation angles in degrees (R = Rz·Ry·Rx), synced to/from `extrinsics.r`.
    manual_euler_deg: [f32; 3],
    // Patterns panel
    patterns: Vec<Pattern>,
    patterns_pending: Rc<RefCell<Option<Vec<Pattern>>>>,
    active_pattern: String,
    active_pattern_pending: Rc<RefCell<Option<String>>>,
    pattern_editing: Option<PatternDraft>,
    pattern_new_name: String,
    // Display settings (LED brightness / gamma). Updates are PUT to the server
    // while the slider is being dragged; server debounces via its DB NOTIFY path.
    settings: SculptureSettings,
    settings_pending: Rc<RefCell<Option<SculptureSettings>>>,
    /// GL-backed renderer for LED cubes (depth-tested). None if glow init failed.
    scene_renderer: Option<Arc<Mutex<SceneRenderer>>>,
}

struct FpsCounter {
    frame_count: u32,
    last_time: f64, // performance.now() in ms
    pub fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        let now = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        Self {
            frame_count: 0,
            last_time: now,
            fps: 0.0,
        }
    }

    fn tick(&mut self) {
        self.frame_count += 1;
        let now = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
        let elapsed = (now - self.last_time) / 1000.0; // convert to seconds
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed as f32;
            self.frame_count = 0;
            self.last_time = now;
        }
    }
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ws_url = get_ws_url();
        let (ws_sender, ws_receiver) =
            ewebsock::connect(&ws_url, ewebsock::Options::default())
                .expect("Failed to connect WebSocket");

        // Fetch LED positions once at startup.
        // Uses Rc<RefCell<...>> (not std::sync::mpsc) since spawn_local is single-threaded WASM.
        let led_pending: Rc<RefCell<Option<Vec<LedPoint>>>> = Rc::new(RefCell::new(None));
        let led_pending_clone = led_pending.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(resp) = gloo_net::http::Request::get("/api/leds").send().await {
                if let Ok(leds) = resp.json::<Vec<LedPoint>>().await {
                    *led_pending_clone.borrow_mut() = Some(leds);
                }
            }
        });

        // Fetch camera serial number.
        let camera_id_pending: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        let cid_clone = camera_id_pending.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(resp) = gloo_net::http::Request::get("/api/calibration/camera_id")
                .send()
                .await
            {
                if let Ok(text) = resp.text().await {
                    *cid_clone.borrow_mut() = Some(text);
                }
            }
        });

        // Fetch existing extrinsics (204 = none stored).
        let extrinsics_pending: Rc<RefCell<Option<Option<CameraExtrinsics>>>> =
            Rc::new(RefCell::new(None));
        let ext_clone = extrinsics_pending.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(resp) = gloo_net::http::Request::get("/api/calibration/extrinsics")
                .send()
                .await
            {
                if resp.status() == 204 {
                    *ext_clone.borrow_mut() = Some(None);
                } else if let Ok(ext) = resp.json::<CameraExtrinsics>().await {
                    *ext_clone.borrow_mut() = Some(Some(ext));
                }
            }
        });

        // Fetch calibration points collected in a previous session on the server.
        let points_pending: Rc<RefCell<Option<Vec<CalibrationPoint>>>> =
            Rc::new(RefCell::new(None));
        let pts_clone = points_pending.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(resp) = gloo_net::http::Request::get("/api/calibration/points")
                .send()
                .await
            {
                if let Ok(pts) = resp.json::<Vec<CalibrationPoint>>().await {
                    *pts_clone.borrow_mut() = Some(pts);
                }
            }
        });

        // Fetch pattern list.
        let patterns_pending: Rc<RefCell<Option<Vec<Pattern>>>> = Rc::new(RefCell::new(None));
        let pp_clone = patterns_pending.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(resp) = gloo_net::http::Request::get("/api/patterns").send().await {
                if let Ok(patterns) = resp.json::<Vec<Pattern>>().await {
                    *pp_clone.borrow_mut() = Some(patterns);
                }
            }
        });

        // Fetch active pattern name.
        let active_pattern_pending: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
        let ap_clone = active_pattern_pending.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(resp) = gloo_net::http::Request::get("/api/patterns/active").send().await {
                if resp.status() == 200 {
                    if let Ok(name) = resp.text().await {
                        *ap_clone.borrow_mut() = Some(name);
                    }
                } else {
                    *ap_clone.borrow_mut() = Some(String::new());
                }
            }
        });

        // Fetch current display settings (brightness, gamma).
        let settings_pending: Rc<RefCell<Option<SculptureSettings>>> = Rc::new(RefCell::new(None));
        let sp_clone = settings_pending.clone();
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(resp) = gloo_net::http::Request::get("/api/settings").send().await {
                if resp.status() == 200 {
                    if let Ok(s) = resp.json::<SculptureSettings>().await {
                        *sp_clone.borrow_mut() = Some(s);
                    }
                }
            }
        });

        Self {
            ws_sender,
            ws_receiver,
            rgb_texture: None,
            ir_texture: None,
            latest_rgb_faces: Vec::new(),
            latest_ir_faces: Vec::new(),
            rgb_size: [640, 480],
            ir_size: [640, 480],
            roi_rect: None,
            tracked_rgb_idx: None,
            tracked_ir_idx: None,
            fps_counter: FpsCounter::new(),
            connected: false,
            local_config: DetectionConfig::default(),
            active_config: DetectionConfig::default(),
            leds: Vec::new(),
            tracking: Vec::new(),
            led_colors: Vec::new(),
            led_pending,
            scene_yaw: 108.0_f32.to_radians(),
            scene_pitch: 12.0_f32.to_radians(),
            scene_zoom: 2.4,
            show_cameras: true,
            show_scene: true,
            show_patterns: true,
            scene_lock: SceneLock::Free,
            tracking_source: TrackingSource::DepthCamera,
            last_virtual_send: 0.0,
            calib_mode: false,
            calib_sub_mode: CalibrationMode::PickPoint,
            calib_points: Vec::new(),
            calib_world_input: [String::new(), String::new(), String::new()],
            calib_pending_cam: None,
            calib_capture_error: Rc::new(RefCell::new(None)),
            extrinsics: None,
            camera_id: String::new(),
            extrinsics_pending,
            points_pending,
            camera_id_pending,
            manual_t: [0.0; 3],
            manual_euler_deg: [0.0; 3],
            patterns: Vec::new(),
            patterns_pending,
            active_pattern: String::new(),
            active_pattern_pending,
            pattern_editing: None,
            pattern_new_name: String::new(),
            settings: SculptureSettings { brightness: 1.0, gamma: 2.2 },
            settings_pending,
            scene_renderer: cc.gl.as_ref().and_then(|gl| {
                match SceneRenderer::new(gl) {
                    Ok(r) => Some(Arc::new(Mutex::new(r))),
                    Err(e) => {
                        tracing::error!("scene3d init failed: {e}");
                        None
                    }
                }
            }),
        }
    }

    fn handle_binary_message(&mut self, ctx: &egui::Context, data: &[u8]) {
        let Some((rgb_jpeg, ir_jpeg, metadata)) = decode_frame_message(data) else {
            return;
        };

        self.latest_rgb_faces = metadata.rgb_faces;
        self.latest_ir_faces = metadata.ir_faces;
        self.rgb_size = metadata.rgb_size;
        self.ir_size = metadata.ir_size;
        self.active_config = metadata.active_config;
        self.roi_rect = metadata.roi_rect;
        self.tracked_rgb_idx = metadata.tracked_rgb_idx;
        self.tracked_ir_idx = metadata.tracked_ir_idx;

        // Decode RGB JPEG
        if let Some(color_image) = decode_jpeg_rgb(rgb_jpeg) {
            match &mut self.rgb_texture {
                Some(tex) => tex.set(color_image, egui::TextureOptions::LINEAR),
                None => {
                    self.rgb_texture = Some(ctx.load_texture(
                        "rgb_feed",
                        color_image,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
        }

        // Decode IR JPEG (grayscale)
        if let Some(color_image) = decode_jpeg_gray(ir_jpeg) {
            match &mut self.ir_texture {
                Some(tex) => tex.set(color_image, egui::TextureOptions::LINEAR),
                None => {
                    self.ir_texture = Some(ctx.load_texture(
                        "ir_feed",
                        color_image,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
        }

        self.fps_counter.tick();
    }

    /// Send the current local_config to the server as a JSON text message.
    fn send_config(&mut self) {
        let msg = ClientMessage::Config(self.local_config);
        if let Ok(json) = serde_json::to_string(&msg) {
            self.ws_sender.send(WsMessage::Text(json));
        }
    }

    /// Tell the server to lock tracking onto the given camera-frame XYZ.
    fn send_select_person(&mut self, xyz: [f32; 3]) {
        let msg = ClientMessage::SelectPerson { xyz };
        if let Ok(json) = serde_json::to_string(&msg) {
            self.ws_sender.send(WsMessage::Text(json));
        }
    }

    /// Set (Some) or clear (None) the server-side virtual tracking override.
    /// Sent on mode-transition and throttled while dragging in Virtual mode.
    fn send_virtual_location(&mut self, xyz: Option<[f32; 3]>) {
        let msg = ClientMessage::VirtualLocation { xyz };
        if let Ok(json) = serde_json::to_string(&msg) {
            self.ws_sender.send(WsMessage::Text(json));
        }
    }

    /// Enable or disable server-side tracking updates to the renderer.
    /// `false` freezes the LED state so the operator can inspect from other angles.
    fn send_tracking_enabled(&mut self, enabled: bool) {
        let msg = ClientMessage::TrackingEnabled(enabled);
        if let Ok(json) = serde_json::to_string(&msg) {
            self.ws_sender.send(WsMessage::Text(json));
        }
    }

    /// Fire-and-forget PUT of the current (brightness, gamma) to the server.
    /// The DB trigger fans out to the renderer, so the LEDs update within a frame.
    fn push_settings(&self) {
        let s = self.settings;
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(req) = gloo_net::http::Request::put("/api/settings").json(&s) {
                let _ = req.send().await;
            }
        });
    }

    /// Switch the active pattern, optimistically updating the local state.
    fn activate_pattern(&mut self, name: String) {
        self.active_pattern = name.clone();
        let ap = self.active_pattern_pending.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let url = format!("/api/patterns/{}/activate", name);
            if let Ok(resp) = gloo_net::http::Request::post(&url).send().await {
                if resp.status() == 204 {
                    *ap.borrow_mut() = Some(name);
                }
            }
        });
    }
}

/// Build a rotation matrix from Euler XYZ angles (degrees).
/// Convention: R = Rz · Ry · Rx  (extrinsic rotations: X first, then Y, then Z).
fn euler_to_rotation(rx_deg: f32, ry_deg: f32, rz_deg: f32) -> [[f32; 3]; 3] {
    let (rx, ry, rz) = (rx_deg.to_radians(), ry_deg.to_radians(), rz_deg.to_radians());
    let (sx, cx) = (rx.sin(), rx.cos());
    let (sy, cy) = (ry.sin(), ry.cos());
    let (sz, cz) = (rz.sin(), rz.cos());
    [
        [ cy*cz,  cz*sx*sy - cx*sz,  cx*cz*sy + sx*sz],
        [ cy*sz,  cx*cz + sx*sy*sz,  cx*sy*sz - cz*sx],
        [-sy,     cy*sx,              cx*cy            ],
    ]
}

/// Extract Euler XYZ angles (degrees) from a rotation matrix (R = Rz·Ry·Rx).
fn rotation_to_euler_deg(r: &[[f32; 3]; 3]) -> [f32; 3] {
    let sy = -r[2][0];
    let ry = sy.clamp(-1.0, 1.0).asin();
    let cy = ry.cos();
    let (rx, rz) = if cy.abs() > 1e-6 {
        (r[2][1].atan2(r[2][2]), r[1][0].atan2(r[0][0]))
    } else {
        // Gimbal lock — fold everything into Rx
        (r[0][1].atan2(r[1][1]), 0.0)
    };
    [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()]
}

fn algo_label(algo: DetectionAlgo) -> &'static str {
    match algo {
        DetectionAlgo::YoloHead => "YOLO Head (fast)",
        DetectionAlgo::ScrfdFace => "SCRFD Face (landmarks)",
        DetectionAlgo::YoloHeadScrfdLandmarks => "YOLO+SCRFD (two-stage)",
    }
}

fn stream_label(s: StreamSelection) -> &'static str {
    match s {
        StreamSelection::Rgb => "RGB",
        StreamSelection::Ir => "IR",
    }
}

fn decode_jpeg_rgb(jpeg_data: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory_with_format(jpeg_data, image::ImageFormat::Jpeg).ok()?;
    let rgb = img.to_rgb8();
    let pixels: Vec<egui::Color32> = rgb
        .pixels()
        .map(|p| egui::Color32::from_rgb(p[0], p[1], p[2]))
        .collect();
    Some(egui::ColorImage {
        size: [rgb.width() as usize, rgb.height() as usize],
        pixels,
    })
}

fn decode_jpeg_gray(jpeg_data: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory_with_format(jpeg_data, image::ImageFormat::Jpeg).ok()?;
    let gray = img.to_luma8();
    let pixels: Vec<egui::Color32> = gray
        .pixels()
        .map(|p| egui::Color32::from_gray(p[0]))
        .collect();
    Some(egui::ColorImage {
        size: [gray.width() as usize, gray.height() as usize],
        pixels,
    })
}


impl eframe::App for App {
    fn on_exit(&mut self, gl: Option<&eframe::glow::Context>) {
        if let (Some(gl), Some(r)) = (gl, self.scene_renderer.as_ref()) {
            if let Ok(r) = r.lock() {
                r.destroy(gl);
            }
        }
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain LED fetch result (arrives once after startup)
        if let Some(leds) = self.led_pending.borrow_mut().take() {
            self.leds = leds;
        }
        // Drain calibration fetch results
        if let Some(id) = self.camera_id_pending.borrow_mut().take() {
            self.camera_id = id;
        }
        if let Some(maybe_ext) = self.extrinsics_pending.borrow_mut().take() {
            self.extrinsics = maybe_ext;
            if let Some(ref ext) = self.extrinsics {
                self.manual_t = ext.t;
                self.manual_euler_deg = rotation_to_euler_deg(&ext.r);
            }
        }
        if let Some(pts) = self.points_pending.borrow_mut().take() {
            self.calib_points = pts;
        }
        if let Some(patterns) = self.patterns_pending.borrow_mut().take() {
            self.patterns = patterns;
        }
        if let Some(active) = self.active_pattern_pending.borrow_mut().take() {
            self.active_pattern = active;
        }
        if let Some(s) = self.settings_pending.borrow_mut().take() {
            self.settings = s;
        }

        // Drain the WebSocket queue, keeping only the newest of each kind.
        // While the tab is hidden egui's rAF pauses but the WebSocket keeps
        // delivering; without this coalescing, the first update() after the tab
        // regains focus would synchronously JPEG-decode the entire backlog.
        let mut latest_binary: Option<Vec<u8>> = None;
        let mut latest_tracking: Option<Vec<TrackingPoint>> = None;
        let mut latest_led_colors: Option<Vec<[u8; 3]>> = None;
        let mut latest_config: Option<DetectionConfig> = None;

        while let Some(event) = self.ws_receiver.try_recv() {
            match event {
                WsEvent::Opened => self.connected = true,
                WsEvent::Closed => self.connected = false,
                WsEvent::Error(e) => {
                    tracing::warn!("WebSocket error: {}", e);
                    self.connected = false;
                }
                WsEvent::Message(WsMessage::Binary(data)) => {
                    latest_binary = Some(data);
                }
                WsEvent::Message(WsMessage::Text(text)) => {
                    if let Ok(msg) = serde_json::from_str::<ServerMessage>(&text) {
                        match msg {
                            ServerMessage::Tracking(pts) => latest_tracking = Some(pts),
                            ServerMessage::LedColors(colors) => latest_led_colors = Some(colors),
                            ServerMessage::Config(cfg) => latest_config = Some(cfg),
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(data) = latest_binary {
            self.handle_binary_message(ctx, &data);
        }
        if let Some(pts) = latest_tracking {
            self.tracking = pts;
        }
        if let Some(colors) = latest_led_colors {
            self.led_colors = colors;
        }
        if let Some(cfg) = latest_config {
            self.local_config = cfg;
            self.active_config = cfg;
        }

        // Top panel with stats and config controls
        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let status = if self.connected { "Connected" } else { "Disconnected" };
                ui.label(format!(
                    "{} | FPS: {:.1} | RGB: {} | IR: {}",
                    status,
                    self.fps_counter.fps,
                    self.latest_rgb_faces.len(),
                    self.latest_ir_faces.len(),
                ));

                ui.separator();

                // Algorithm selector
                ui.label("Algo:");
                let prev_algo = self.local_config.algo;
                egui::ComboBox::from_id_salt("algo_select")
                    .selected_text(algo_label(self.local_config.algo))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.local_config.algo,
                            DetectionAlgo::YoloHead,
                            algo_label(DetectionAlgo::YoloHead),
                        );
                        ui.selectable_value(
                            &mut self.local_config.algo,
                            DetectionAlgo::ScrfdFace,
                            algo_label(DetectionAlgo::ScrfdFace),
                        );
                        ui.selectable_value(
                            &mut self.local_config.algo,
                            DetectionAlgo::YoloHeadScrfdLandmarks,
                            algo_label(DetectionAlgo::YoloHeadScrfdLandmarks),
                        );
                    });
                if self.local_config.algo != prev_algo {
                    self.send_config();
                }

                ui.separator();

                // Stream selector
                ui.label("Stream:");
                let prev_stream = self.local_config.stream;
                egui::ComboBox::from_id_salt("stream_select")
                    .selected_text(stream_label(self.local_config.stream))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.local_config.stream,
                            StreamSelection::Rgb,
                            stream_label(StreamSelection::Rgb),
                        );
                        ui.selectable_value(
                            &mut self.local_config.stream,
                            StreamSelection::Ir,
                            stream_label(StreamSelection::Ir),
                        );
                    });
                if self.local_config.stream != prev_stream {
                    self.send_config();
                }

                // Show server-side active config as confirmation (may lag one frame)
                if self.active_config != self.local_config {
                    ui.separator();
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        format!(
                            "Server: {} / {}",
                            algo_label(self.active_config.algo),
                            stream_label(self.active_config.stream),
                        ),
                    );
                }
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_cameras, "Cameras");
                ui.checkbox(&mut self.show_scene, "3D Scene");
                ui.checkbox(&mut self.show_patterns, "Patterns");

                ui.separator();

                // Quick pattern switcher — lets the user flip between patterns
                // without opening the Patterns window.
                ui.label("Pattern:");
                let enabled_patterns: Vec<String> = self
                    .patterns
                    .iter()
                    .filter(|p| p.enabled)
                    .map(|p| p.name.clone())
                    .collect();
                let current_label = if self.active_pattern.is_empty() {
                    "(none)".to_string()
                } else {
                    self.active_pattern.clone()
                };
                let mut new_active: Option<String> = None;
                egui::ComboBox::from_id_salt("quick_pattern_select")
                    .selected_text(current_label)
                    .show_ui(ui, |ui| {
                        for name in &enabled_patterns {
                            let selected = name == &self.active_pattern;
                            if ui.selectable_label(selected, name).clicked() && !selected {
                                new_active = Some(name.clone());
                            }
                        }
                    });
                if let Some(name) = new_active {
                    self.activate_pattern(name);
                }

                ui.separator();

                // Brightness / gamma sliders. Push on every change — the server
                // clamps and the renderer picks it up through the DB NOTIFY path.
                ui.label("Brightness:");
                let b_resp = ui.add(
                    egui::Slider::new(&mut self.settings.brightness, 0.0..=2.0)
                        .fixed_decimals(2),
                );
                let g_resp = ui.add(
                    egui::Slider::new(&mut self.settings.gamma, 0.5..=3.0)
                        .text("γ")
                        .fixed_decimals(2),
                );
                if b_resp.changed() || g_resp.changed() {
                    self.push_settings();
                }
            });
        });

        // Background panel (egui requires a CentralPanel)
        egui::CentralPanel::default().show(ctx, |_ui| {});

        // ── Cameras window ────────────────────────────────────────
        // Clicks on face boxes flow out of the closure via these locals, so we can
        // call `self.send_select_person` after the `open: &mut self.show_cameras`
        // borrow is released.
        let mut rgb_selection: Option<[f32; 3]> = None;
        let mut ir_selection: Option<[f32; 3]> = None;
        egui::Window::new("Cameras")
            .default_size([640.0, 400.0])
            .default_pos([10.0, 80.0])
            .open(&mut self.show_cameras)
            .show(ctx, |ui| {
                    let show_rgb = matches!(self.active_config.stream, StreamSelection::Rgb);
                    let show_ir = matches!(self.active_config.stream, StreamSelection::Ir);
                    let num_panels = show_rgb as u32 + show_ir as u32;

                    ui.horizontal(|ui| {
                        let available = ui.available_size();
                        let panel_width = if num_panels > 0 {
                            (available.x - 10.0 * (num_panels - 1) as f32) / num_panels as f32
                        } else {
                            available.x
                        };

                        if show_rgb {
                            ui.vertical(|ui| {
                                ui.label("RGB + Face");
                                if let Some(tex) = &self.rgb_texture {
                                    let aspect = self.rgb_size[1] as f32 / self.rgb_size[0] as f32;
                                    let display_h = panel_width * aspect;
                                    let display_size = egui::vec2(panel_width, display_h);

                                    let (rect, _response) =
                                        ui.allocate_exact_size(display_size, egui::Sense::hover());

                                    ui.painter().image(
                                        tex.id(),
                                        rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );

                                    let scale_x = rect.width() / self.rgb_size[0] as f32;
                                    let scale_y = rect.height() / self.rgb_size[1] as f32;
                                    let clicked = overlay::draw_faces(
                                        ui,
                                        &self.latest_rgb_faces,
                                        rect.left_top(),
                                        scale_x,
                                        scale_y,
                                        self.tracked_rgb_idx,
                                        "rgb_face",
                                    );
                                    overlay::draw_roi(
                                        ui.painter(),
                                        self.roi_rect,
                                        rect.left_top(),
                                        scale_x,
                                        scale_y,
                                    );
                                    if let Some(i) = clicked {
                                        if let Some(xyz) = self
                                            .latest_rgb_faces
                                            .get(i)
                                            .and_then(|f| f.xyz)
                                        {
                                            rgb_selection = Some(xyz);
                                        }
                                    }
                                } else {
                                    ui.label("Waiting for camera...");
                                }
                            });

                            if show_ir {
                                ui.separator();
                            }
                        }

                        if show_ir {
                            ui.vertical(|ui| {
                                ui.label("IR + Face");
                                if let Some(tex) = &self.ir_texture {
                                    let aspect = self.ir_size[1] as f32 / self.ir_size[0] as f32;
                                    let display_h = panel_width * aspect;
                                    let display_size = egui::vec2(panel_width, display_h);

                                    let (rect, _response) =
                                        ui.allocate_exact_size(display_size, egui::Sense::hover());

                                    ui.painter().image(
                                        tex.id(),
                                        rect,
                                        egui::Rect::from_min_max(
                                            egui::pos2(0.0, 0.0),
                                            egui::pos2(1.0, 1.0),
                                        ),
                                        egui::Color32::WHITE,
                                    );

                                    let scale_x = rect.width() / self.ir_size[0] as f32;
                                    let scale_y = rect.height() / self.ir_size[1] as f32;
                                    let clicked = overlay::draw_faces(
                                        ui,
                                        &self.latest_ir_faces,
                                        rect.left_top(),
                                        scale_x,
                                        scale_y,
                                        self.tracked_ir_idx,
                                        "ir_face",
                                    );
                                    overlay::draw_roi(
                                        ui.painter(),
                                        self.roi_rect,
                                        rect.left_top(),
                                        scale_x,
                                        scale_y,
                                    );
                                    if let Some(i) = clicked {
                                        if let Some(xyz) = self
                                            .latest_ir_faces
                                            .get(i)
                                            .and_then(|f| f.xyz)
                                        {
                                            ir_selection = Some(xyz);
                                        }
                                    }
                                } else {
                                    ui.label("Waiting for camera...");
                                }
                            });
                        }
                    });
            });

        // If the user clicked a face box, forward the camera-frame XYZ to the server.
        // IR takes precedence when both fire in the same frame (rare; only one panel
        // is visible at a time given the current stream selection).
        if let Some(xyz) = ir_selection.or(rgb_selection) {
            self.send_select_person(xyz);
        }

        // ── 3D Scene window ───────────────────────────────────────
        // Virtual-location and tracking-enabled sends happen after the Window
        // closure returns so the `&mut self.show_scene` borrow taken by `.open(...)`
        // doesn't conflict with the `&mut self` the helpers need.
        // Outer option = "has a send"; inner option = Some(xyz) to set, None to clear.
        let mut pending_virtual_send: Option<Option<[f32; 3]>> = None;
        let mut pending_tracking_enabled: Option<bool> = None;

        egui::Window::new("3D Scene")
            .default_size([600.0, 500.0])
            .default_pos([660.0, 80.0])
            .open(&mut self.show_scene)
            .vscroll(false)
            .show(ctx, |ui| {
                    // ── Calibration side panel ────────────────────────────────
                    egui::SidePanel::right("calib_panel")
                        .resizable(false)
                        .default_width(210.0)
                        .show_inside(ui, |ui| {
                            ui.heading("Calibration");
                            let cam_label = if self.camera_id.is_empty() {
                                "…".to_string()
                            } else {
                                self.camera_id.clone()
                            };
                            ui.small(format!("Camera: {}", cam_label));

                            // Scene camera driver — free orbit or a specific tracked person.
                            ui.separator();
                            ui.label("Lock to:");
                            let selected_label = match &self.scene_lock {
                                SceneLock::Free => "Free".to_string(),
                                SceneLock::Tracked(name) => {
                                    name.chars().take(8).collect::<String>()
                                }
                            };
                            egui::ComboBox::from_id_salt("location_lock")
                                .selected_text(selected_label)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.scene_lock,
                                        SceneLock::Free,
                                        "Free",
                                    );
                                    let names: Vec<String> =
                                        self.tracking.iter().map(|p| p.name.clone()).collect();
                                    for name in names {
                                        let label = name.chars().take(8).collect::<String>();
                                        ui.selectable_value(
                                            &mut self.scene_lock,
                                            SceneLock::Tracked(name.clone()),
                                            label,
                                        );
                                    }
                                });

                            // Tracking source — independent of the scene camera.
                            // Controls what the LED renderer's `location` uniform follows.
                            ui.add_space(4.0);
                            ui.label("Tracking source:");
                            let prev_source = self.tracking_source;
                            egui::ComboBox::from_id_salt("tracking_source")
                                .selected_text(match self.tracking_source {
                                    TrackingSource::DepthCamera => "Depth camera",
                                    TrackingSource::SceneCamera => "Scene camera",
                                    TrackingSource::Frozen => "Frozen",
                                })
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut self.tracking_source,
                                        TrackingSource::DepthCamera,
                                        "Depth camera",
                                    );
                                    ui.selectable_value(
                                        &mut self.tracking_source,
                                        TrackingSource::SceneCamera,
                                        "Scene camera",
                                    );
                                    ui.selectable_value(
                                        &mut self.tracking_source,
                                        TrackingSource::Frozen,
                                        "Frozen",
                                    );
                                });
                            // Transition side-effects on the server's tracking inputs:
                            //   * Entering Frozen  → TrackingEnabled(false)
                            //   * Leaving  Frozen  → TrackingEnabled(true)
                            //   * Leaving  SceneCamera to DepthCamera → clear virtual override
                            //   * SceneCamera → Frozen deliberately keeps the override
                            //     so the last streamed position stays the frozen state.
                            if prev_source != self.tracking_source {
                                match (prev_source, self.tracking_source) {
                                    (_, TrackingSource::Frozen) => {
                                        pending_tracking_enabled = Some(false);
                                    }
                                    (TrackingSource::Frozen, _) => {
                                        pending_tracking_enabled = Some(true);
                                    }
                                    _ => {}
                                }
                                if prev_source == TrackingSource::SceneCamera
                                    && self.tracking_source == TrackingSource::DepthCamera
                                {
                                    pending_virtual_send = Some(None);
                                }
                            }

                            ui.separator();

                            ui.checkbox(&mut self.calib_mode, "Calibration mode");
                            if self.calib_mode {
                                ui.horizontal(|ui| {
                                    ui.radio_value(
                                        &mut self.calib_sub_mode,
                                        CalibrationMode::PickPoint,
                                        "Pick point",
                                    );
                                    ui.radio_value(
                                        &mut self.calib_sub_mode,
                                        CalibrationMode::StandInView,
                                        "Stand in view",
                                    );
                                });
                                match self.calib_sub_mode {
                                    CalibrationMode::PickPoint => {
                                        ui.colored_label(
                                            egui::Color32::YELLOW,
                                            "Click a tracking point\nin the scene to capture it",
                                        );
                                    }
                                    CalibrationMode::StandInView => {
                                        ui.colored_label(
                                            egui::Color32::YELLOW,
                                            "Stand at a known spot,\nalign via translation sliders,\nthen capture.",
                                        );
                                    }
                                }
                            }

                            ui.separator();

                            // Status
                            if self.extrinsics.is_some() {
                                ui.colored_label(egui::Color32::GREEN, "Extrinsics active");
                            } else {
                                ui.colored_label(egui::Color32::from_gray(160), "Uncalibrated");
                            }

                            ui.add_space(4.0);
                            ui.label(format!("{} pair(s) collected", self.calib_points.len()));

                            ui.separator();
                            match self.calib_sub_mode {
                                CalibrationMode::PickPoint => {
                                    ui.label("World XYZ (m, Y+ up, pixo frame):");
                                    ui.horizontal(|ui| {
                                        for (hint, val) in ["X", "Y", "Z"]
                                            .iter()
                                            .zip(self.calib_world_input.iter_mut())
                                        {
                                            ui.add(
                                                egui::TextEdit::singleline(val)
                                                    .hint_text(*hint)
                                                    .desired_width(52.0),
                                            );
                                        }
                                    });

                                    ui.add_space(4.0);
                                    if let Some(cam) = self.calib_pending_cam {
                                        ui.label(format!(
                                            "Cam: ({:.3}, {:.3}, {:.3})",
                                            cam[0], cam[1], cam[2]
                                        ));

                                        let can_add = self.calib_world_input.iter().all(|s| {
                                            s.trim().parse::<f32>().is_ok()
                                        });
                                        if ui
                                            .add_enabled(can_add, egui::Button::new("Add pair"))
                                            .clicked()
                                        {
                                            let wx = self.calib_world_input[0]
                                                .trim()
                                                .parse::<f32>()
                                                .unwrap();
                                            let wy = self.calib_world_input[1]
                                                .trim()
                                                .parse::<f32>()
                                                .unwrap();
                                            let wz = self.calib_world_input[2]
                                                .trim()
                                                .parse::<f32>()
                                                .unwrap();
                                            let pt = CalibrationPoint {
                                                cam,
                                                world: [wx, wy, wz],
                                            };
                                            self.calib_points.push(pt);
                                            self.calib_pending_cam = None;
                                            wasm_bindgen_futures::spawn_local(async move {
                                                let _ = gloo_net::http::Request::post(
                                                    "/api/calibration/points",
                                                )
                                                .json(&pt)
                                                .expect("serialize CalibrationPoint")
                                                .send()
                                                .await;
                                            });
                                        }
                                        if ui.small_button("Cancel").clicked() {
                                            self.calib_pending_cam = None;
                                        }
                                    } else {
                                        ui.colored_label(
                                            egui::Color32::from_gray(140),
                                            "No cam point selected",
                                        );
                                    }
                                }
                                CalibrationMode::StandInView => {
                                    let tracked = self.tracked_rgb_idx.is_some()
                                        || self.tracked_ir_idx.is_some();
                                    if tracked {
                                        ui.colored_label(egui::Color32::GREEN, "Tracked");
                                    } else {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(220, 90, 90),
                                            "Not tracked",
                                        );
                                    }

                                    if ui
                                        .add_enabled(tracked, egui::Button::new("Capture pair"))
                                        .clicked()
                                    {
                                        let err_cell = self.calib_capture_error.clone();
                                        let points_cell = self.points_pending.clone();
                                        wasm_bindgen_futures::spawn_local(async move {
                                            match gloo_net::http::Request::post(
                                                "/api/calibration/capture_tracked",
                                            )
                                            .send()
                                            .await
                                            {
                                                Ok(resp) if resp.status() == 200 => {
                                                    *err_cell.borrow_mut() = None;
                                                    // Re-fetch the canonical list from the server
                                                    // so our in-memory mirror stays in sync.
                                                    if let Ok(r) = gloo_net::http::Request::get(
                                                        "/api/calibration/points",
                                                    )
                                                    .send()
                                                    .await
                                                    {
                                                        if let Ok(pts) = r
                                                            .json::<Vec<CalibrationPoint>>()
                                                            .await
                                                        {
                                                            *points_cell.borrow_mut() = Some(pts);
                                                        }
                                                    }
                                                }
                                                Ok(resp) => {
                                                    *err_cell.borrow_mut() = Some(format!(
                                                        "Capture failed ({})",
                                                        resp.status()
                                                    ));
                                                }
                                                Err(_) => {
                                                    *err_cell.borrow_mut() =
                                                        Some("Capture request failed".into());
                                                }
                                            }
                                        });
                                    }

                                    if let Some(msg) = self.calib_capture_error.borrow().as_ref() {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(220, 90, 90),
                                            msg,
                                        );
                                    }

                                    ui.small(
                                        "After moving, click your\nface in the video to\nre-lock tracking.",
                                    );
                                }
                            }

                            ui.separator();

                            let can_compute = self.calib_points.len() >= 3;
                            if ui
                                .add_enabled(can_compute, egui::Button::new("Compute extrinsics"))
                                .clicked()
                            {
                                let ext_pending = self.extrinsics_pending.clone();
                                wasm_bindgen_futures::spawn_local(async move {
                                    if let Ok(resp) =
                                        gloo_net::http::Request::post("/api/calibration/compute")
                                            .send()
                                            .await
                                    {
                                        if let Ok(ext) =
                                            resp.json::<CameraExtrinsics>().await
                                        {
                                            *ext_pending.borrow_mut() = Some(Some(ext));
                                        }
                                    }
                                });
                            }

                            if ui.button("Clear pairs").clicked() {
                                self.calib_points.clear();
                                self.calib_pending_cam = None;
                                wasm_bindgen_futures::spawn_local(async move {
                                    let _ = gloo_net::http::Request::delete(
                                        "/api/calibration/points",
                                    )
                                    .send()
                                    .await;
                                });
                            }

                            if self.extrinsics.is_some()
                                && ui.button("Clear extrinsics").clicked()
                            {
                                self.extrinsics = None;
                                self.manual_t = [0.0; 3];
                                self.manual_euler_deg = [0.0; 3];
                                wasm_bindgen_futures::spawn_local(async move {
                                    let _ = gloo_net::http::Request::delete(
                                        "/api/calibration/extrinsics",
                                    )
                                    .send()
                                    .await;
                                });
                            }

                            // ── Manual adjust ────────────────────────────────
                            ui.separator();
                            ui.label("Manual adjust:");

                            ui.label("Translation (m, Y+ up):");
                            let mut t_changed = false;
                            ui.horizontal(|ui| {
                                t_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut self.manual_t[0])
                                            .speed(0.005)
                                            .prefix("X: ")
                                            .fixed_decimals(3),
                                    )
                                    .changed();
                                t_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut self.manual_t[1])
                                            .speed(0.005)
                                            .prefix("Y: ")
                                            .fixed_decimals(3),
                                    )
                                    .changed();
                                t_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut self.manual_t[2])
                                            .speed(0.005)
                                            .prefix("Z: ")
                                            .fixed_decimals(3),
                                    )
                                    .changed();
                            });

                            let mut r_changed = false;
                            if self.calib_sub_mode == CalibrationMode::PickPoint {
                                ui.label("Rotation (°):");
                                ui.horizontal(|ui| {
                                    r_changed |= ui
                                        .add(
                                            egui::DragValue::new(&mut self.manual_euler_deg[0])
                                                .speed(0.2)
                                                .prefix("X: ")
                                                .fixed_decimals(1),
                                        )
                                        .changed();
                                    r_changed |= ui
                                        .add(
                                            egui::DragValue::new(&mut self.manual_euler_deg[1])
                                                .speed(0.2)
                                                .prefix("Y: ")
                                                .fixed_decimals(1),
                                        )
                                        .changed();
                                    r_changed |= ui
                                        .add(
                                            egui::DragValue::new(&mut self.manual_euler_deg[2])
                                                .speed(0.2)
                                                .prefix("Z: ")
                                                .fixed_decimals(1),
                                        )
                                        .changed();
                                });
                            }

                            if t_changed || r_changed {
                                let r = euler_to_rotation(
                                    self.manual_euler_deg[0],
                                    self.manual_euler_deg[1],
                                    self.manual_euler_deg[2],
                                );
                                let new_ext = CameraExtrinsics { r, t: self.manual_t };
                                self.extrinsics = Some(new_ext);
                                wasm_bindgen_futures::spawn_local(async move {
                                    let _ = gloo_net::http::Request::put(
                                        "/api/calibration/extrinsics",
                                    )
                                    .json(&new_ext)
                                    .expect("serialize CameraExtrinsics")
                                    .send()
                                    .await;
                                });
                            }
                        });

                    // ── 3D scatter view ────────────────────────────────────────
                    // Tracked mode auto-follows the person; Free and Virtual
                    // use the same orbit controls (drag = rotate, scroll = zoom).
                    // Virtual additionally streams the orbit eye position to the
                    // server as a virtual tracking location (handled below).
                    let is_locked = if let SceneLock::Tracked(locked_name) = &self.scene_lock {
                        if let Some(pt) = self.tracking.iter().find(|p| &p.name == locked_name)
                        {
                            let dx = pt.x;
                            let dy = pt.y;
                            let dz = pt.z;
                            let r_xz = (dx * dx + dz * dz).sqrt();
                            self.scene_yaw = dz.atan2(dx);
                            self.scene_pitch = dy.atan2(r_xz);
                            self.scene_zoom = (dx * dx + dy * dy + dz * dz).sqrt().max(0.5);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    let view_height = (ui.available_height() - 4.0).max(160.0);
                    let sense = if is_locked {
                        egui::Sense::hover()
                    } else {
                        egui::Sense::click_and_drag()
                    };
                    let (rect, response) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), view_height),
                        sense,
                    );

                    if !is_locked && response.dragged() {
                        let d = response.drag_delta();
                        self.scene_yaw += d.x * 0.005;
                        self.scene_pitch =
                            (self.scene_pitch + d.y * 0.005).clamp(-1.5, 1.5);
                    }
                    if !is_locked && response.hovered() {
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                        self.scene_zoom = (self.scene_zoom - scroll * 0.01).clamp(0.5, 20.0);
                    }

                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(15, 15, 25));

                    let hint = if self.calib_mode {
                        "3D View  (calibration mode — click a tracking point)"
                    } else if is_locked {
                        "3D View  (locked to location)"
                    } else {
                        match self.tracking_source {
                            TrackingSource::SceneCamera => {
                                "3D View  (drag = orbit  |  scroll = zoom  |  tracking → scene camera)"
                            }
                            TrackingSource::Frozen => {
                                "3D View  (drag = orbit  |  scroll = zoom  |  tracking frozen)"
                            }
                            TrackingSource::DepthCamera => {
                                "3D View  (drag = orbit  |  scroll = zoom)"
                            }
                        }
                    };
                    painter.text(
                        rect.left_top() + egui::vec2(6.0, 4.0),
                        egui::Align2::LEFT_TOP,
                        hint,
                        egui::FontId::proportional(11.0),
                        egui::Color32::from_gray(100),
                    );

                    // World frame = pixo: X=right, Y=up, Z=forward (right-handed OpenGL).
                    // Orbit camera: eye = Zoom * (cos(yaw)cos(pitch), sin(pitch), sin(yaw)cos(pitch)),
                    // looking at origin with WorldUp = (0, 1, 0). Transcribed from pixo's IsoCamera.
                    let cx = rect.center().x;
                    let cy = rect.center().y;
                    let scale = 500.0_f32; // pixels per metre at unit depth
                    let (yaw, pitch, dist) =
                        (self.scene_yaw, self.scene_pitch, self.scene_zoom);
                    let (cp, sp) = (pitch.cos(), pitch.sin());
                    let (cyw, sy) = (yaw.cos(), yaw.sin());
                    let eye = [dist * cyw * cp, dist * sp, dist * sy * cp];

                    // Scene-camera tracking: stream the orbit eye as the tracking
                    // location so the LED renderer reacts as if a person were here.
                    // Throttled to ~30 Hz to match the drag/orbit update cadence.
                    if self.tracking_source == TrackingSource::SceneCamera {
                        let now = ctx.input(|i| i.time);
                        if now - self.last_virtual_send > 0.03 {
                            pending_virtual_send = Some(Some(eye));
                            self.last_virtual_send = now;
                        }
                    }

                    let fwd = [-cyw * cp, -sp, -sy * cp]; // unit vector from eye to origin
                    // right = normalize(fwd × WorldUp(0,1,0)) = normalize((-fwd.z, 0, fwd.x));
                    // up = right × fwd.
                    let rx0 = -fwd[2];
                    let rz0 = fwd[0];
                    let rlen = (rx0 * rx0 + rz0 * rz0).sqrt().max(1e-6);
                    let right = [rx0 / rlen, 0.0_f32, rz0 / rlen];
                    let up = [
                        right[1] * fwd[2] - right[2] * fwd[1],
                        right[2] * fwd[0] - right[0] * fwd[2],
                        right[0] * fwd[1] - right[1] * fwd[0],
                    ];

                    let project = |px: f32, py: f32, pz: f32| -> Option<egui::Pos2> {
                        let d = [px - eye[0], py - eye[1], pz - eye[2]];
                        let vx = d[0] * right[0] + d[1] * right[1] + d[2] * right[2];
                        let vy = d[0] * up[0] + d[1] * up[1] + d[2] * up[2];
                        let vz = d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2];
                        if vz < 0.01 {
                            return None;
                        }
                        // Screen y grows downward; world +Y (up) → negative screen-y offset.
                        Some(egui::pos2(cx + scale * vx / vz, cy - scale * vy / vz))
                    };

                    // LEDs — rendered as real 5mm cubes in a GL pass with a depth buffer
                    // so closer LEDs correctly occlude farther ones. Projection matches the
                    // 2D `project()` closure above (fovy = 2·atan(H/1000), scale = 500 px/m)
                    // so tracking dots and camera axes overlay-align with the LED cubes.
                    if let Some(renderer) = self.scene_renderer.clone() {
                        let has_colors = self.led_colors.len() == self.leds.len()
                            && !self.leds.is_empty();
                        let mut inst: Vec<f32> = Vec::with_capacity(self.leds.len() * 6);
                        for (i, led) in self.leds.iter().enumerate() {
                            let (r, g, b) = if has_colors {
                                let [r, g, b] = self.led_colors[i];
                                (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
                            } else {
                                (60.0 / 255.0, 60.0 / 255.0, 60.0 / 255.0)
                            };
                            inst.extend_from_slice(&[led.x, led.y, led.z, r, g, b]);
                        }
                        if !inst.is_empty() {
                            let view = scene3d::look_at(
                                eye,
                                [0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                            );
                            let rw = rect.width().max(1.0);
                            let rh = rect.height().max(1.0);
                            let fovy = 2.0 * (rh / 1000.0).atan();
                            let aspect = rw / rh;
                            let proj = scene3d::perspective(fovy, aspect, 0.01, 100.0);
                            let mvp = scene3d::mul_mat4(&proj, &view);
                            let half_m = 0.0025_f32;
                            let brightness = self.settings.brightness;
                            let gamma = self.settings.gamma;
                            let cb = egui_glow::CallbackFn::new(move |info, painter| {
                                let vp = info.viewport_in_pixels();
                                if let Ok(mut r) = renderer.lock() {
                                    r.paint(
                                        painter.gl(),
                                        (vp.left_px, vp.from_bottom_px, vp.width_px, vp.height_px),
                                        &mvp,
                                        half_m,
                                        brightness,
                                        gamma,
                                        &inst,
                                    );
                                }
                            });
                            painter.add(egui::PaintCallback {
                                rect,
                                callback: Arc::new(cb),
                            });
                        }
                    }

                    // Tracking locations — larger cyan dots with short UUID label
                    for tp in &self.tracking {
                        let is_locked_tp = matches!(
                            &self.scene_lock,
                            SceneLock::Tracked(n) if n == &tp.name
                        );
                        let is_pending = self
                            .calib_pending_cam
                            .map(|p| p[0] == tp.x && p[1] == tp.y && p[2] == tp.z)
                            .unwrap_or(false);
                        if let Some(p) = project(tp.x, tp.y, tp.z) {
                            if rect.contains(p) {
                                let color = if is_pending {
                                    egui::Color32::from_rgb(255, 100, 255)
                                } else if is_locked_tp {
                                    egui::Color32::from_rgb(255, 180, 0)
                                } else {
                                    egui::Color32::from_rgb(0, 220, 180)
                                };
                                painter.circle_filled(p, 6.0, color);
                                painter.circle_stroke(
                                    p,
                                    6.0,
                                    egui::Stroke::new(1.5, egui::Color32::WHITE),
                                );
                                let short_name: String = tp.name.chars().take(8).collect();
                                painter.text(
                                    p + egui::vec2(9.0, -6.0),
                                    egui::Align2::LEFT_TOP,
                                    &short_name,
                                    egui::FontId::monospace(9.0),
                                    color,
                                );
                            }
                        }
                    }

                    // Camera position, orientation axes, and frustum — always shown.
                    // When no extrinsics: camera sits at origin and camera frame equals world frame.
                    let identity_ext = CameraExtrinsics {
                        r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        t: [0.0, 0.0, 0.0],
                    };
                    let ext = self.extrinsics.unwrap_or(identity_ext);
                    let [tx, ty, tz] = ext.t;
                    if let Some(origin_px) = project(tx, ty, tz) {
                        let axis_len: f32 = 0.3; // metres
                        // Camera-local axis tips pushed through ext.apply land in world frame.
                        let axes: [([f32; 3], egui::Color32); 3] = [
                            ([axis_len, 0.0, 0.0], egui::Color32::from_rgb(255, 60, 60)), // X red
                            ([0.0, axis_len, 0.0], egui::Color32::from_rgb(60, 220, 60)), // Y green
                            ([0.0, 0.0, axis_len], egui::Color32::from_rgb(60, 100, 255)), // Z blue
                        ];
                        for (dir_cam, color) in axes {
                            let w = ext.apply(dir_cam);
                            if let Some(tip) = project(w[0], w[1], w[2]) {
                                painter.line_segment(
                                    [origin_px, tip],
                                    egui::Stroke::new(2.0, color),
                                );
                            }
                        }

                        // Frustum — D435 at 640×480: fx≈615, fy≈615
                        // tan(half_fov_h) = 320/615 ≈ 0.520
                        // tan(half_fov_v) = 240/615 ≈ 0.390
                        let frust_depth: f32 = 2.0; // metres to far plane
                        let hx = 0.520 * frust_depth;
                        let hy = 0.390 * frust_depth;
                        // Corners in camera-local frame (X right, Y down, Z forward). ext.apply
                        // lifts them into the Y-up world frame.
                        let corners_cam: [[f32; 3]; 4] = [
                            [ hx,  hy, frust_depth], // bottom-right
                            [ hx, -hy, frust_depth], // top-right
                            [-hx, -hy, frust_depth], // top-left
                            [-hx,  hy, frust_depth], // bottom-left
                        ];
                        let frust_color =
                            egui::Color32::from_rgba_unmultiplied(120, 190, 255, 90);
                        let frust_stroke = egui::Stroke::new(1.0, frust_color);
                        let corners_px: Vec<Option<egui::Pos2>> = corners_cam
                            .iter()
                            .map(|&c| {
                                let w = ext.apply(c);
                                project(w[0], w[1], w[2])
                            })
                            .collect();
                        // 4 rays: origin → corner
                        for &cp in &corners_px {
                            if let Some(cp) = cp {
                                painter.line_segment([origin_px, cp], frust_stroke);
                            }
                        }
                        // Far-plane rectangle: 0-1-2-3-0
                        for i in 0..4 {
                            if let (Some(a), Some(b)) =
                                (corners_px[i], corners_px[(i + 1) % 4])
                            {
                                painter.line_segment([a, b], frust_stroke);
                            }
                        }

                        // Origin dot + label
                        let cam_dot_color = if self.extrinsics.is_some() {
                            egui::Color32::WHITE
                        } else {
                            egui::Color32::from_gray(180)
                        };
                        painter.circle_filled(origin_px, 5.0, cam_dot_color);
                        painter.text(
                            origin_px + egui::vec2(7.0, -7.0),
                            egui::Align2::LEFT_BOTTOM,
                            "CAM",
                            egui::FontId::monospace(9.0),
                            cam_dot_color,
                        );
                    }

                    // Calibration: capture a tracking point on click (pick-point mode only)
                    if self.calib_mode
                        && self.calib_sub_mode == CalibrationMode::PickPoint
                        && response.clicked()
                    {
                        if let Some(pos) = response.interact_pointer_pos() {
                            for tp in &self.tracking {
                                if let Some(p) = project(tp.x, tp.y, tp.z) {
                                    if (p - pos).length() < 12.0 {
                                        self.calib_pending_cam = Some([tp.x, tp.y, tp.z]);
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    painter.text(
                        rect.left_bottom() + egui::vec2(6.0, -4.0),
                        egui::Align2::LEFT_BOTTOM,
                        format!(
                            "LEDs: {}  |  Tracking: {}",
                            self.leds.len(),
                            self.tracking.len()
                        ),
                        egui::FontId::proportional(11.0),
                        egui::Color32::from_gray(100),
                    );
            });

        // Drain the virtual-location and tracking-enabled sends deferred from
        // inside the Scene window. Ordering matters on Off → non-Virtual: send
        // enable=true *before* the virtual-clear so the freeze isn't lifted
        // with a stale override still in place on the server.
        if let Some(enabled) = pending_tracking_enabled {
            self.send_tracking_enabled(enabled);
        }
        if let Some(xyz) = pending_virtual_send {
            self.send_virtual_location(xyz);
        }

        // ── Patterns window ───────────────────────────────────────
        egui::Window::new("Patterns")
            .default_size([700.0, 500.0])
            .default_pos([10.0, 490.0])
            .open(&mut self.show_patterns)
            .show(ctx, |ui| {
                    // ── Collect deferred actions (avoids borrow conflicts) ─────
                    let mut select_idx: Option<usize> = None;
                    let mut create_new: Option<String> = None; // name for a brand-new draft
                    let mut delete_name: Option<String> = None;
                    let mut save_pattern: Option<Pattern> = None; // POST new
                    let mut save_update: Option<(String, PatternUpdate)> = None; // PUT existing
                    let mut activate_name: Option<String> = None;

                    // ── Left panel: pattern list ───────────────────────────────
                    egui::SidePanel::left("patterns_list")
                        .resizable(false)
                        .default_width(200.0)
                        .show_inside(ui, |ui| {
                            ui.heading("Patterns");
                            ui.separator();

                            let list_height = (ui.available_height() - 52.0).max(40.0);
                            egui::ScrollArea::vertical()
                                .id_salt("patterns_scroll")
                                .max_height(list_height)
                                .show(ui, |ui| {
                                    for (i, pattern) in self.patterns.iter().enumerate() {
                                        let is_active = pattern.name == self.active_pattern;
                                        let is_selected = self
                                            .pattern_editing
                                            .as_ref()
                                            .map(|d| d.original_name == pattern.name)
                                            .unwrap_or(false);

                                        ui.horizontal(|ui| {
                                            // Active indicator dot doubles as a one-click
                                            // activate button for inactive enabled patterns.
                                            let dot_color = if is_active {
                                                egui::Color32::from_rgb(80, 220, 100)
                                            } else {
                                                egui::Color32::from_gray(55)
                                            };
                                            let dot = egui::Button::new(
                                                egui::RichText::new("●").color(dot_color),
                                            )
                                            .frame(false);
                                            let can_activate = pattern.enabled && !is_active;
                                            let dot_resp = ui
                                                .add_enabled(can_activate, dot)
                                                .on_hover_text(if is_active {
                                                    "Active"
                                                } else if pattern.enabled {
                                                    "Click to activate"
                                                } else {
                                                    "Disabled"
                                                });
                                            if dot_resp.clicked() {
                                                activate_name = Some(pattern.name.clone());
                                            }

                                            let text_color = if !pattern.enabled {
                                                egui::Color32::from_gray(110)
                                            } else if is_active {
                                                egui::Color32::from_rgb(80, 220, 100)
                                            } else {
                                                egui::Color32::WHITE
                                            };
                                            let label = egui::RichText::new(&pattern.name)
                                                .color(text_color);
                                            if ui.selectable_label(is_selected, label).clicked() {
                                                select_idx = Some(i);
                                            }
                                        });
                                    }
                                });

                            ui.separator();

                            // Create new pattern
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.pattern_new_name)
                                        .hint_text("new name…")
                                        .desired_width(110.0),
                                );
                                let can_create =
                                    is_valid_pattern_name(self.pattern_new_name.trim());
                                if ui
                                    .add_enabled(can_create, egui::Button::new("New"))
                                    .clicked()
                                {
                                    create_new =
                                        Some(self.pattern_new_name.trim().to_string());
                                }
                            });
                        });

                    // ── Editor (remaining space) ───────────────────────────────
                    if let Some(ref mut draft) = self.pattern_editing {
                        ui.add_space(4.0);

                        // Header row
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            if draft.is_new {
                                let name_valid = is_valid_pattern_name(draft.name.trim());
                                let te = egui::TextEdit::singleline(&mut draft.name)
                                    .desired_width(160.0);
                                let resp = ui.add(te);
                                if !name_valid && resp.lost_focus() {
                                    ui.colored_label(
                                        egui::Color32::RED,
                                        "letters, digits, _ or - only",
                                    );
                                }
                            } else {
                                ui.label(
                                    egui::RichText::new(&draft.original_name)
                                        .strong()
                                        .monospace(),
                                );
                                if self.active_pattern == draft.original_name {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(80, 220, 100),
                                        "(active)",
                                    );
                                }
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.checkbox(&mut draft.enabled, "Enabled");
                            ui.checkbox(&mut draft.overscan, "Overscan");
                        });

                        ui.separator();
                        ui.label(
                            egui::RichText::new("GLSL Fragment Shader")
                                .small()
                                .color(egui::Color32::from_gray(160)),
                        );

                        // Code editor — fills space above the button row
                        let code_height = (ui.available_height() - 42.0).max(60.0);
                        egui::ScrollArea::vertical()
                            .id_salt("glsl_edit_scroll")
                            .max_height(code_height)
                            .show(ui, |ui| {
                                ui.add(
                                    egui::TextEdit::multiline(&mut draft.glsl)
                                        .font(egui::TextStyle::Monospace)
                                        .desired_width(f32::INFINITY)
                                        .desired_rows(24),
                                );
                            });

                        ui.separator();

                        // Action buttons
                        ui.horizontal(|ui| {
                            let save_label = if draft.is_new { "Create" } else { "Save" };
                            let can_save = if draft.is_new {
                                is_valid_pattern_name(draft.name.trim())
                            } else {
                                true
                            };
                            if ui.add_enabled(can_save, egui::Button::new(save_label)).clicked()
                            {
                                if draft.is_new {
                                    save_pattern = Some(Pattern {
                                        name: draft.name.trim().to_string(),
                                        glsl_code: draft.glsl.clone(),
                                        enabled: draft.enabled,
                                        overscan: draft.overscan,
                                    });
                                    // Transition to edit mode for the just-created pattern.
                                    draft.original_name = draft.name.trim().to_string();
                                    draft.is_new = false;
                                } else {
                                    save_update = Some((
                                        draft.original_name.clone(),
                                        PatternUpdate {
                                            glsl_code: draft.glsl.clone(),
                                            enabled: draft.enabled,
                                            overscan: draft.overscan,
                                        },
                                    ));
                                }
                            }

                            if !draft.is_new {
                                let is_active = self.active_pattern == draft.original_name;
                                if is_active {
                                    ui.colored_label(
                                        egui::Color32::from_rgb(80, 220, 100),
                                        "● Active",
                                    );
                                } else if ui.button("Activate").clicked() {
                                    activate_name = Some(draft.original_name.clone());
                                }

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .button(
                                                egui::RichText::new("Delete")
                                                    .color(egui::Color32::from_rgb(220, 80, 80)),
                                            )
                                            .clicked()
                                        {
                                            delete_name =
                                                Some(draft.original_name.clone());
                                        }
                                    },
                                );
                            }
                        });
                    } else {
                        ui.vertical_centered(|ui| {
                            ui.add_space(80.0);
                            ui.label(
                                egui::RichText::new("Select a pattern or create a new one")
                                    .color(egui::Color32::from_gray(100)),
                            );
                        });
                    }

                    // ── Apply deferred actions ─────────────────────────────────
                    if let Some(i) = select_idx {
                        let p = &self.patterns[i];
                        self.pattern_editing = Some(PatternDraft {
                            original_name: p.name.clone(),
                            name: p.name.clone(),
                            glsl: p.glsl_code.clone(),
                            enabled: p.enabled,
                            overscan: p.overscan,
                            is_new: false,
                        });
                    }

                    if let Some(name) = create_new {
                        self.pattern_editing = Some(PatternDraft {
                            original_name: name.clone(),
                            name: name.clone(),
                            glsl: DEFAULT_PATTERN_GLSL.to_string(),
                            enabled: true,
                            overscan: false,
                            is_new: true,
                        });
                        self.pattern_new_name = String::new();
                    }

                    if let Some(name) = delete_name {
                        self.pattern_editing = None;
                        let pp = self.patterns_pending.clone();
                        let ap = self.active_pattern_pending.clone();
                        wasm_bindgen_futures::spawn_local(async move {
                            let url = format!("/api/patterns/{}", name);
                            if let Ok(resp) =
                                gloo_net::http::Request::delete(&url).send().await
                            {
                                if resp.status() == 204 {
                                    if let Ok(r) =
                                        gloo_net::http::Request::get("/api/patterns")
                                            .send()
                                            .await
                                    {
                                        if let Ok(list) = r.json::<Vec<Pattern>>().await {
                                            *pp.borrow_mut() = Some(list);
                                        }
                                    }
                                    if let Ok(r) =
                                        gloo_net::http::Request::get("/api/patterns/active")
                                            .send()
                                            .await
                                    {
                                        if r.status() == 200 {
                                            if let Ok(active) = r.text().await {
                                                *ap.borrow_mut() = Some(active);
                                            }
                                        } else {
                                            *ap.borrow_mut() = Some(String::new());
                                        }
                                    }
                                }
                            }
                        });
                    }

                    if let Some(pattern) = save_pattern {
                        let pp = self.patterns_pending.clone();
                        wasm_bindgen_futures::spawn_local(async move {
                            if let Ok(resp) = gloo_net::http::Request::post("/api/patterns")
                                .json(&pattern)
                                .expect("serialize Pattern")
                                .send()
                                .await
                            {
                                if resp.status() == 201 || resp.status() == 200 {
                                    if let Ok(r) =
                                        gloo_net::http::Request::get("/api/patterns")
                                            .send()
                                            .await
                                    {
                                        if let Ok(list) = r.json::<Vec<Pattern>>().await {
                                            *pp.borrow_mut() = Some(list);
                                        }
                                    }
                                }
                            }
                        });
                    }

                    if let Some((name, update)) = save_update {
                        let pp = self.patterns_pending.clone();
                        wasm_bindgen_futures::spawn_local(async move {
                            let url = format!("/api/patterns/{}", name);
                            if let Ok(resp) = gloo_net::http::Request::put(&url)
                                .json(&update)
                                .expect("serialize PatternUpdate")
                                .send()
                                .await
                            {
                                if resp.status() == 204 {
                                    if let Ok(r) =
                                        gloo_net::http::Request::get("/api/patterns")
                                            .send()
                                            .await
                                    {
                                        if let Ok(list) = r.json::<Vec<Pattern>>().await {
                                            *pp.borrow_mut() = Some(list);
                                        }
                                    }
                                }
                            }
                        });
                    }

                    if let Some(name) = activate_name {
                        let ap = self.active_pattern_pending.clone();
                        wasm_bindgen_futures::spawn_local(async move {
                            let url = format!("/api/patterns/{}/activate", name);
                            if let Ok(resp) =
                                gloo_net::http::Request::post(&url).send().await
                            {
                                if resp.status() == 204 {
                                    *ap.borrow_mut() = Some(name);
                                }
                            }
                        });
                    }
            });

        // Repaint at ~30 fps (matches camera input rate; avoids rendering
        // all three windows at 60 fps which is unnecessarily expensive).
        ctx.request_repaint_after(std::time::Duration::from_millis(33));
    }
}
