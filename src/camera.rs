use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use realsense_rust::{
    config::Config,
    context::Context as RsContext,
    frame::{ColorFrame, InfraredFrame},
    kind::{Rs2CameraInfo, Rs2Format, Rs2StreamKind},
    pipeline::InactivePipeline,
};
use std::collections::HashSet;

pub struct Camera {
    pipeline: realsense_rust::pipeline::ActivePipeline,
}

pub struct CameraFrames {
    pub rgb: RgbImage,
    pub ir: GrayImage,
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

        let pipeline = pipeline
            .start(Some(config))
            .context("Failed to start pipeline")?;

        Ok(Camera { pipeline })
    }

    /// Try to capture a frame. Returns None on timeout (caller should retry).
    pub fn capture(&mut self) -> Result<Option<CameraFrames>> {
        let timeout = std::time::Duration::from_millis(500);
        let frames = match self.pipeline.wait(Some(timeout)) {
            Ok(f) => f,
            Err(_) => return Ok(None), // timeout, not an error
        };

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

        let rgb = rgb_image.unwrap_or_else(|| RgbImage::new(rgb_w, rgb_h));
        let ir = ir_image.unwrap_or_else(|| GrayImage::new(640, 480));

        Ok(Some(CameraFrames {
            rgb,
            ir,
        }))
    }
}
