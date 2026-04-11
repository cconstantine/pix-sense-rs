mod overlay;

use eframe::egui;
use ewebsock::{WsEvent, WsMessage, WsReceiver, WsSender};
use pix_sense_common::{
    decode_frame_message, DetectionAlgo, DetectionConfig, FaceDetection, StreamSelection,
};
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

struct App {
    ws_sender: WsSender,
    ws_receiver: WsReceiver,
    rgb_texture: Option<egui::TextureHandle>,
    ir_texture: Option<egui::TextureHandle>,
    depth_texture: Option<egui::TextureHandle>,
    latest_rgb_faces: Vec<FaceDetection>,
    latest_ir_faces: Vec<FaceDetection>,
    rgb_size: [u32; 2],
    ir_size: [u32; 2],
    depth_size: [u32; 2],
    fps_counter: FpsCounter,
    connected: bool,
    /// Config the user has selected in the UI (sent to server on change).
    local_config: DetectionConfig,
    /// Config the server reports as active (echoed back in FrameMetadata).
    active_config: DetectionConfig,
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
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let ws_url = get_ws_url();
        let (ws_sender, ws_receiver) =
            ewebsock::connect(&ws_url, ewebsock::Options::default())
                .expect("Failed to connect WebSocket");

        Self {
            ws_sender,
            ws_receiver,
            rgb_texture: None,
            ir_texture: None,
            depth_texture: None,
            latest_rgb_faces: Vec::new(),
            latest_ir_faces: Vec::new(),
            rgb_size: [640, 480],
            ir_size: [640, 480],
            depth_size: [640, 480],
            fps_counter: FpsCounter::new(),
            connected: false,
            local_config: DetectionConfig::default(),
            active_config: DetectionConfig::default(),
        }
    }

    fn handle_binary_message(&mut self, ctx: &egui::Context, data: &[u8]) {
        let Some((rgb_jpeg, ir_jpeg, depth_jpeg, metadata)) = decode_frame_message(data) else {
            return;
        };

        self.latest_rgb_faces = metadata.rgb_faces;
        self.latest_ir_faces = metadata.ir_faces;
        self.rgb_size = metadata.rgb_size;
        self.ir_size = metadata.ir_size;
        self.depth_size = metadata.depth_size;
        self.active_config = metadata.active_config;

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

        // Decode depth JPEG (grayscale) and colorize client-side
        if let Some(color_image) = decode_jpeg_depth(depth_jpeg) {
            match &mut self.depth_texture {
                Some(tex) => tex.set(color_image, egui::TextureOptions::LINEAR),
                None => {
                    self.depth_texture = Some(ctx.load_texture(
                        "depth_feed",
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
        if let Ok(json) = serde_json::to_string(&self.local_config) {
            self.ws_sender.send(WsMessage::Text(json));
        }
    }
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
        StreamSelection::Both => "RGB + IR",
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

fn decode_jpeg_depth(jpeg_data: &[u8]) -> Option<egui::ColorImage> {
    let img = image::load_from_memory_with_format(jpeg_data, image::ImageFormat::Jpeg).ok()?;
    let gray = img.to_luma8();
    let pixels: Vec<egui::Color32> = gray
        .pixels()
        .map(|p| depth_colormap(p[0]))
        .collect();
    Some(egui::ColorImage {
        size: [gray.width() as usize, gray.height() as usize],
        pixels,
    })
}

/// Piecewise-linear colormap: 0 → black (no data), near → blue, mid → green, far → red.
fn depth_colormap(v: u8) -> egui::Color32 {
    if v == 0 {
        return egui::Color32::BLACK;
    }
    // Map 1..=255 into 0.0..=1.0
    let t = (v - 1) as f32 / 254.0;
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll WebSocket for new frames
        while let Some(event) = self.ws_receiver.try_recv() {
            match event {
                WsEvent::Opened => {
                    self.connected = true;
                    // Send initial config so server immediately uses what's shown in UI
                    self.send_config();
                }
                WsEvent::Message(WsMessage::Binary(data)) => {
                    self.handle_binary_message(ctx, &data);
                }
                WsEvent::Error(e) => {
                    tracing::warn!("WebSocket error: {}", e);
                    self.connected = false;
                }
                WsEvent::Closed => {
                    self.connected = false;
                }
                _ => {}
            }
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
                        ui.selectable_value(
                            &mut self.local_config.stream,
                            StreamSelection::Both,
                            stream_label(StreamSelection::Both),
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
        });

        // Central panel with camera feeds
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                let available = ui.available_size();
                let panel_width = (available.x - 20.0) / 3.0; // 3 feeds with separators

                // RGB feed with face overlay
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
                        overlay::draw_faces(
                            ui.painter(),
                            &self.latest_rgb_faces,
                            rect.left_top(),
                            scale_x,
                            scale_y,
                        );
                    } else {
                        ui.label("Waiting for camera...");
                    }
                });

                ui.separator();

                // IR feed with face overlay
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
                        overlay::draw_faces(
                            ui.painter(),
                            &self.latest_ir_faces,
                            rect.left_top(),
                            scale_x,
                            scale_y,
                        );
                    } else {
                        ui.label("Waiting for camera...");
                    }
                });

                ui.separator();

                // Depth feed
                ui.vertical(|ui| {
                    ui.label("Depth");
                    if let Some(tex) = &self.depth_texture {
                        let aspect = self.depth_size[1] as f32 / self.depth_size[0] as f32;
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
                    } else {
                        ui.label("Waiting for camera...");
                    }
                });
            });
        });

        // Request continuous repaint for live video
        ctx.request_repaint();
    }
}
