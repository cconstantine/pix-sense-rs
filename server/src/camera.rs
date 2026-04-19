use anyhow::{Context, Result};
use image::{GrayImage, RgbImage};
use realsense_rust::{
    config::Config,
    context::Context as RsContext,
    frame::{ColorFrame, DepthFrame, InfraredFrame},
    kind::{Rs2CameraInfo, Rs2Format, Rs2Option, Rs2StreamKind},
    pipeline::InactivePipeline,
    processing_blocks::{
        align::Align,
        hole_filling::HoleFillingFilter,
        temporal_filter::TemporalFilter,
    },
};
use std::collections::HashSet;
use std::ffi::CString;

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
    /// Typical Intel RealSense D435 values at 848×480. Used as a fallback only.
    fn default() -> Self {
        Self { fx: 607.0, fy: 607.0, ppx: 424.0, ppy: 240.0 }
    }
}

pub struct Camera {
    pipeline: realsense_rust::pipeline::ActivePipeline,
    align: Align,
    intrinsics: CameraIntrinsics,
    temporal_filter: TemporalFilter,
    hole_filling: HoleFillingFilter,
}

pub struct CameraFrames {
    pub rgb: RgbImage,
    pub ir: GrayImage,
    pub depth_raw: Vec<u16>, // Z16 values in mm, same W×H as depth
    pub depth_size: [u32; 2],
    pub intrinsics: CameraIntrinsics,
}

/// One RealSense device detected at enumeration time. The `device` handle is
/// discarded on open — we bind the pipeline to the device via its serial.
#[derive(Debug, Clone)]
pub struct EnumeratedDevice {
    pub serial: String,
    pub name: String,
}

/// Enumerate all connected RealSense devices without starting any pipeline.
/// Returns an empty vec if no devices are found (callers may still want to
/// serve non-camera routes, e.g. patterns/settings).
pub fn enumerate() -> Result<Vec<EnumeratedDevice>> {
    let context = RsContext::new().context("Failed to create RealSense context")?;
    let devices = context.query_devices(HashSet::new());
    let mut out = Vec::with_capacity(devices.len());
    for device in devices.iter() {
        let serial = device
            .info(Rs2CameraInfo::SerialNumber)
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let name = device
            .info(Rs2CameraInfo::Name)
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        if serial.is_empty() {
            tracing::warn!("Skipping RealSense device with missing serial ({})", name);
            continue;
        }
        out.push(EnumeratedDevice { serial, name });
    }
    Ok(out)
}

impl Camera {
    /// Open a pipeline bound to the device with the given serial.
    pub fn new(device: &EnumeratedDevice) -> Result<Self> {
        let context = RsContext::new().context("Failed to create RealSense context")?;

        tracing::info!(
            "Opening RealSense device: {} {}",
            device.name, device.serial
        );

        let pipeline =
            InactivePipeline::try_from(&context).context("Failed to create pipeline")?;

        let mut config = Config::new();
        let serial_c = CString::new(device.serial.as_bytes())
            .context("Camera serial contains interior nul byte")?;
        config
            .enable_device_from_serial(&serial_c)
            .context("Failed to bind pipeline to camera serial")?;
        // 848×480 keeps three D435s within USB 3.0 bandwidth on a single host
        // controller. 1280×720 on all three streams saturates the bus and two
        // cameras hang in pipeline.start() waiting for frames that never arrive.
        config
            .enable_stream(Rs2StreamKind::Color, None, 848, 480, Rs2Format::Rgb8, 30)
            .context("Failed to enable color stream")?;
        config
            .enable_stream(Rs2StreamKind::Infrared, None, 848, 480, Rs2Format::Y8, 30)
            .context("Failed to enable infrared stream")?;
        config
            .enable_stream(Rs2StreamKind::Depth, None, 848, 480, Rs2Format::Z16, 30)
            .context("Failed to enable depth stream")?;

        let pipeline = pipeline
            .start(Some(config))
            .context("Failed to start pipeline")?;

        // Enable the IR emitter (dot projector) for improved depth quality
        for mut sensor in pipeline.profile().device().sensors() {
            if sensor.set_option(Rs2Option::EmitterEnabled, 1.0).is_ok() {
                tracing::info!("Enabled IR emitter (dot projector)");
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

        let temporal_filter = TemporalFilter::new(10)
            .context("Failed to create temporal filter")?;
        let hole_filling = HoleFillingFilter::new(10)
            .context("Failed to create hole-filling filter")?;
        tracing::info!("Depth post-processing filters enabled (temporal + hole-filling)");

        Ok(Camera { pipeline, align, intrinsics, temporal_filter, hole_filling })
    }

    /// Try to capture a frame. Returns None on timeout (caller should retry).
    pub fn capture(&mut self) -> Result<Option<CameraFrames>> {
        let timeout = std::time::Duration::from_millis(500);
        let raw_frames = match self.pipeline.wait(Some(timeout)) {
            Ok(f) => f,
            Err(_) => return Ok(None), // timeout, not an error
        };

        // Align all streams to the colour camera's coordinate space
        self.align.queue(raw_frames)
            .context("Failed to queue frames for alignment")?;
        let frames = self.align.wait(std::time::Duration::from_millis(100))
            .context("Failed to get aligned frames")?;

        let mut rgb_image = None;
        let mut ir_image = None;
        let mut rgb_w = 1280u32;
        let mut rgb_h = 720u32;

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
        // Apply temporal filter (smooths across frames) then hole-filling before
        // reading raw u16 values for 3-D deprojection.
        let mut depth_raw_buf: Vec<u16> = Vec::new();
        let mut depth_w = 1280u32;
        let mut depth_h = 720u32;
        let filter_timeout = std::time::Duration::from_millis(100);

        // Pipe: raw depth → temporal filter → hole-filling → extract u16 data.
        // Each filter's queue() consumes the frame, so we need ownership via into_iter().
        let filtered_depth = frames
            .frames_of_type::<DepthFrame>()
            .into_iter()
            .next()
            .and_then(|depth| {
                self.temporal_filter.queue(depth).ok()?;
                let depth = self.temporal_filter.wait(filter_timeout).ok()?;
                self.hole_filling.queue(depth).ok()?;
                self.hole_filling.wait(filter_timeout).ok()
            });

        if let Some(depth) = filtered_depth.as_ref() {
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
            }
        }

        let rgb = rgb_image.unwrap_or_else(|| RgbImage::new(rgb_w, rgb_h));
        let ir = ir_image.unwrap_or_else(|| GrayImage::new(rgb_w, rgb_h));

        Ok(Some(CameraFrames {
            rgb,
            ir,
            depth_raw: depth_raw_buf,
            depth_size: [depth_w, depth_h],
            intrinsics: self.intrinsics,
        }))
    }
}

