mod camera;
mod face;
mod overlay;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

use anyhow::Result;
use eframe::egui;
use image::{GrayImage, RgbImage};

use camera::Camera;
use face::{FaceDetection, FaceDetector};

const MODEL_PATH: &str = "models/scrfd_10g_bnkps.onnx";

/// Data sent from the processing thread to the UI thread
struct FrameData {
    rgb: RgbImage,
    ir: GrayImage,
    rgb_faces: Vec<FaceDetection>,
    ir_faces: Vec<FaceDetection>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    // Verify model exists
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("ERROR: Model not found at '{}'", MODEL_PATH);
        eprintln!("Run ./setup.sh to download the SCRFD model.");
        std::process::exit(1);
    }

    // Channel for sending frames from processing thread to UI
    let (tx, rx) = mpsc::sync_channel::<FrameData>(2);
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Spawn camera + face detection thread
    let handle = thread::spawn(move || {
        if let Err(e) = processing_thread(tx, &running_clone) {
            tracing::error!("Processing thread error: {:#}", e);
        }
    });

    // Run the GUI
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1320.0, 540.0])
            .with_title("pix-sense-rs"),
        ..Default::default()
    };

    eframe::run_native(
        "pix-sense-rs",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc, rx)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;

    // Signal processing thread to stop and wait for clean shutdown
    running.store(false, Ordering::Relaxed);
    let _ = handle.join();

    Ok(())
}

fn processing_thread(tx: mpsc::SyncSender<FrameData>, running: &AtomicBool) -> Result<()> {
    // Load model first — CUDA graph compilation can take a while,
    // and the RealSense pipeline may stall if frames aren't consumed.
    tracing::info!("Loading face detection model...");
    let mut detector = FaceDetector::new(MODEL_PATH)?;
    tracing::info!("Face detector ready");

    tracing::info!("Initializing camera...");
    let mut camera = Camera::new()?;
    tracing::info!("Camera initialized");

    while running.load(Ordering::Relaxed) {
        let frames = match camera.capture() {
            Ok(Some(f)) => f,
            Ok(None) => continue, // timeout, retry
            Err(e) => {
                tracing::warn!("Camera capture error: {:#}", e);
                continue;
            }
        };

        let rgb_faces = match detector.detect_rgb(&frames.rgb) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("RGB face detection error: {:#}", e);
                Vec::new()
            }
        };

        let ir_faces = match detector.detect_gray(&frames.ir) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("IR face detection error: {:#}", e);
                Vec::new()
            }
        };

        let data = FrameData {
            rgb: frames.rgb,
            ir: frames.ir,
            rgb_faces,
            ir_faces,
        };

        // If the UI is behind, drop frames rather than building up latency
        if tx.try_send(data).is_err() {
            tracing::trace!("Frame dropped (UI too slow)");
        }
    }

    tracing::info!("Processing thread shutting down");
    Ok(())
}

struct App {
    rx: mpsc::Receiver<FrameData>,
    rgb_texture: Option<egui::TextureHandle>,
    ir_texture: Option<egui::TextureHandle>,
    latest_rgb_faces: Vec<FaceDetection>,
    latest_ir_faces: Vec<FaceDetection>,
    rgb_size: [u32; 2],
    ir_size: [u32; 2],
    fps_counter: FpsCounter,
}

struct FpsCounter {
    frame_count: u32,
    last_time: std::time::Instant,
    fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frame_count: 0,
            last_time: std::time::Instant::now(),
            fps: 0.0,
        }
    }

    fn tick(&mut self) {
        self.frame_count += 1;
        let elapsed = self.last_time.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed;
            self.frame_count = 0;
            self.last_time = std::time::Instant::now();
        }
    }
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>, rx: mpsc::Receiver<FrameData>) -> Self {
        Self {
            rx,
            rgb_texture: None,
            ir_texture: None,
            latest_rgb_faces: Vec::new(),
            latest_ir_faces: Vec::new(),
            rgb_size: [640, 480],
            ir_size: [640, 480],
            fps_counter: FpsCounter::new(),
        }
    }

    fn update_textures(&mut self, ctx: &egui::Context, data: &FrameData) {
        // Update RGB texture
        let rgb_pixels: Vec<egui::Color32> = data
            .rgb
            .pixels()
            .map(|p| egui::Color32::from_rgb(p[0], p[1], p[2]))
            .collect();

        let rgb_image = egui::ColorImage {
            size: [data.rgb.width() as usize, data.rgb.height() as usize],
            pixels: rgb_pixels,
        };

        self.rgb_size = [data.rgb.width(), data.rgb.height()];

        match &mut self.rgb_texture {
            Some(tex) => tex.set(rgb_image, egui::TextureOptions::LINEAR),
            None => {
                self.rgb_texture = Some(ctx.load_texture(
                    "rgb_feed",
                    rgb_image,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }

        // Update IR texture (grayscale -> RGB for display)
        let ir_pixels: Vec<egui::Color32> = data
            .ir
            .pixels()
            .map(|p| egui::Color32::from_gray(p[0]))
            .collect();

        let ir_image = egui::ColorImage {
            size: [data.ir.width() as usize, data.ir.height() as usize],
            pixels: ir_pixels,
        };

        self.ir_size = [data.ir.width(), data.ir.height()];

        match &mut self.ir_texture {
            Some(tex) => tex.set(ir_image, egui::TextureOptions::LINEAR),
            None => {
                self.ir_texture = Some(ctx.load_texture(
                    "ir_feed",
                    ir_image,
                    egui::TextureOptions::LINEAR,
                ));
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain channel, keeping only the latest frame
        let mut latest = None;
        while let Ok(data) = self.rx.try_recv() {
            latest = Some(data);
        }

        if let Some(data) = latest {
            self.latest_rgb_faces = data.rgb_faces.clone();
            self.latest_ir_faces = data.ir_faces.clone();
            self.update_textures(ctx, &data);
            self.fps_counter.tick();
        }

        // Top panel with stats
        egui::TopBottomPanel::top("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "FPS: {:.1} | RGB Faces: {} | IR Faces: {}",
                    self.fps_counter.fps,
                    self.latest_rgb_faces.len(),
                    self.latest_ir_faces.len()
                ));
            });
        });

        // Central panel with camera feeds
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // RGB feed with face overlay
                ui.vertical(|ui| {
                    ui.label("RGB + Face");
                    if let Some(tex) = &self.rgb_texture {
                        let available = ui.available_size();
                        let half_width = (available.x - 10.0) / 2.0;
                        let aspect = self.rgb_size[1] as f32 / self.rgb_size[0] as f32;
                        let display_h = half_width * aspect;
                        let display_size = egui::vec2(half_width, display_h);

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
                        let available = ui.available_size();
                        let half_width = (available.x - 10.0).max(100.0);
                        let aspect = self.ir_size[1] as f32 / self.ir_size[0] as f32;
                        let display_h = half_width * aspect;
                        let display_size = egui::vec2(half_width, display_h);

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
            });
        });

        // Request continuous repaint for live video
        ctx.request_repaint();
    }
}
