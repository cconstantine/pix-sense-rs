use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use realsense_rust::{
    config::Config,
    context::Context as RsContext,
    frame::{ColorFrame, DepthFrame, InfraredFrame},
    kind::{Rs2CameraInfo, Rs2Format, Rs2Option, Rs2StreamKind},
    pipeline::InactivePipeline,
    processing_blocks::align::Align,
};
use std::collections::HashSet;

/// Pinhole camera intrinsics for the colour stream (all streams are aligned to it).
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    /// Principal point X (cx)
    pub ppx: f32,
    /// Principal point Y (cy)
    pub ppy: f32,
}

impl Default for CameraIntrinsics {
    /// Typical Intel RealSense D435 values at 640×480. Used as a fallback only.
    fn default() -> Self {
        Self { fx: 615.0, fy: 615.0, ppx: 320.0, ppy: 240.0 }
    }
}

pub struct Camera {
    pipeline: realsense_rust::pipeline::ActivePipeline,
    align: Align,
    intrinsics: CameraIntrinsics,
}

pub struct CameraFrames {
    pub rgb: RgbImage,
    pub ir: GrayImage,
    pub depth: GrayImage,
    pub depth_raw: Vec<u16>, // Z16 values in mm, same W×H as depth
    pub depth_size: [u32; 2],
    pub intrinsics: CameraIntrinsics,
}

/// Query the serial number of the first connected RealSense device without
/// starting the full pipeline. Returns `None` if no device is found.
pub fn query_serial() -> Option<String> {
    let context = RsContext::new().ok()?;
    let devices = context.query_devices(HashSet::new());
    let device = devices.first()?;
    Some(
        device
            .info(Rs2CameraInfo::SerialNumber)
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
    )
}

impl Camera {
    pub fn new() -> Result<Self> {
        let context = RsContext::new().context("Failed to create RealSense context")?;

        let devices = context.query_devices(HashSet::new());
        if devices.len() == 0 {
            anyhow::bail!("No RealSense devices found. Is the camera connected?");
        }

        let device = &devices[0];
        tracing::info!(
            "Found RealSense device: {} {}",
            device
                .info(Rs2CameraInfo::Name)
                .unwrap_or_default()
                .to_string_lossy(),
            device
                .info(Rs2CameraInfo::SerialNumber)
                .unwrap_or_default()
                .to_string_lossy(),
        );

        let pipeline =
            InactivePipeline::try_from(&context).context("Failed to create pipeline")?;

        let mut config = Config::new();
        config
            .enable_stream(Rs2StreamKind::Color, None, 640, 480, Rs2Format::Rgb8, 30)
            .context("Failed to enable color stream")?;
        config
            .enable_stream(Rs2StreamKind::Infrared, None, 640, 480, Rs2Format::Y8, 30)
            .context("Failed to enable infrared stream")?;
        config
            .enable_stream(Rs2StreamKind::Depth, None, 640, 480, Rs2Format::Z16, 30)
            .context("Failed to enable depth stream")?;

        let pipeline = pipeline
            .start(Some(config))
            .context("Failed to start pipeline")?;

        // Disable the IR dot projector so it doesn't interfere with the IR image
        for mut sensor in pipeline.profile().device().sensors() {
            if sensor.set_option(Rs2Option::EmitterEnabled, 0.0).is_ok() {
                tracing::info!("Disabled IR emitter (dot projector)");
            }
        }

        // Extract colour-stream intrinsics for 3-D deprojection.
        let intrinsics = pipeline
            .profile()
            .streams()
            .iter()
            .find(|s| s.kind() == Rs2StreamKind::Color)
            .and_then(|s| s.intrinsics().ok())
            .map(|i| CameraIntrinsics { fx: i.fx(), fy: i.fy(), ppx: i.ppx(), ppy: i.ppy() })
            .unwrap_or_else(|| {
                tracing::warn!("Could not read camera intrinsics — using D435 defaults");
                CameraIntrinsics::default()
            });
        tracing::info!(
            "Camera intrinsics: fx={:.1} fy={:.1} cx={:.1} cy={:.1}",
            intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        );

        // Align IR and depth streams to the color camera's coordinate space.
        // This keeps RGB native (no reprojection artifacts) and reprojects IR to match.
        let align = Align::new(Rs2StreamKind::Color, 10)
            .context("Failed to create alignment processing block")?;

        Ok(Camera { pipeline, align, intrinsics })
    }

    /// Try to capture a frame. Returns None on timeout (caller should retry).
    pub fn capture(&mut self) -> Result<Option<CameraFrames>> {
        let timeout = std::time::Duration::from_millis(500);
        let raw_frames = match self.pipeline.wait(Some(timeout)) {
            Ok(f) => f,
            Err(_) => return Ok(None), // timeout, not an error
        };

        // Align all streams to the infrared coordinate space
        self.align.queue(raw_frames)
            .context("Failed to queue frames for alignment")?;
        let frames = self.align.wait(std::time::Duration::from_millis(100))
            .context("Failed to get aligned frames")?;

        let mut rgb_image = None;
        let mut ir_image = None;
        let mut rgb_w = 640u32;
        let mut rgb_h = 480u32;

        // Extract color frames using raw data for speed
        let color_frames = frames.frames_of_type::<ColorFrame>();
        if let Some(color) = color_frames.first() {
            let w = color.width() as u32;
            let h = color.height() as u32;
            rgb_w = w;
            rgb_h = h;

            let data_size = color.get_data_size();
            let expected = (w * h * 3) as usize;
            if data_size >= expected {
                let raw = unsafe {
                    std::slice::from_raw_parts(
                        color.get_data() as *const std::os::raw::c_void as *const u8,
                        data_size,
                    )
                };
                if let Some(img) = RgbImage::from_raw(w, h, raw[..expected].to_vec()) {
                    rgb_image = Some(img);
                }
            }
        }

        // Extract infrared frames using raw data for speed
        let ir_frames = frames.frames_of_type::<InfraredFrame>();
        if let Some(ir) = ir_frames.first() {
            let w = ir.width() as u32;
            let h = ir.height() as u32;

            let data_size = ir.get_data_size();
            let expected = (w * h) as usize;
            if data_size >= expected {
                let raw = unsafe {
                    std::slice::from_raw_parts(
                        ir.get_data() as *const std::os::raw::c_void as *const u8,
                        data_size,
                    )
                };
                if let Some(img) = GrayImage::from_raw(w, h, raw[..expected].to_vec()) {
                    ir_image = Some(img);
                }
            }
        }

        // Extract depth frames (Z16: 16-bit unsigned, millimeters)
        // Keep raw u16 values for 3-D deprojection; also produce 8-bit grayscale for display.
        let mut depth_image = None;
        let mut depth_raw_buf: Vec<u16> = Vec::new();
        let mut depth_w = 640u32;
        let mut depth_h = 480u32;
        let depth_frames = frames.frames_of_type::<DepthFrame>();
        if let Some(depth) = depth_frames.first() {
            let w = depth.width() as u32;
            let h = depth.height() as u32;
            depth_w = w;
            depth_h = h;

            let data_size = depth.get_data_size();
            let expected = (w * h * 2) as usize; // Z16 = 2 bytes per pixel
            if data_size >= expected {
                let raw = unsafe {
                    std::slice::from_raw_parts(
                        depth.get_data() as *const std::os::raw::c_void as *const u16,
                        (w * h) as usize,
                    )
                };
                depth_raw_buf = raw.to_vec();
                depth_image = Some(depth_to_gray(raw, w, h));
            }
        }

        let rgb = rgb_image.unwrap_or_else(|| RgbImage::new(rgb_w, rgb_h));
        let ir = ir_image.unwrap_or_else(|| GrayImage::new(640, 480));
        let depth = depth_image.unwrap_or_else(|| GrayImage::new(depth_w, depth_h));

        Ok(Some(CameraFrames {
            rgb,
            ir,
            depth,
            depth_raw: depth_raw_buf,
            depth_size: [depth_w, depth_h],
            intrinsics: self.intrinsics,
        }))
    }
}

/// Map Z16 depth to 8-bit grayscale using a fixed range (300–5000 mm).
/// 0 → black (no data), near → bright, far → dark.
fn depth_to_gray(data: &[u16], w: u32, h: u32) -> GrayImage {
    const MIN_MM: u16 = 300;
    const MAX_MM: u16 = 5000;
    const RANGE: f32 = (MAX_MM - MIN_MM) as f32;

    let mut pixels = Vec::with_capacity((w * h) as usize);
    for &d in data {
        let v = if d == 0 || d < MIN_MM {
            0u8
        } else if d >= MAX_MM {
            255
        } else {
            ((d - MIN_MM) as f32 / RANGE * 255.0) as u8
        };
        pixels.push(v);
    }
    GrayImage::from_raw(w, h, pixels).unwrap_or_else(|| GrayImage::new(w, h))
}
