use anyhow::{Context as _, Result};
use glutin::api::egl::{
    context::PossiblyCurrentContext,
    device::Device,
    display::Display,
    surface::Surface,
};
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, Version};
use glutin::display::GlDisplay;
use glutin::surface::{PbufferSurface, SurfaceAttributesBuilder};
use std::num::NonZeroU32;

/// A headless OpenGL ES 3.0 context created via EGL without a window or display server.
///
/// Uses a 1×1 Pbuffer surface as the draw surface — all actual rendering goes to
/// framebuffer objects (FBOs).  The NVIDIA EGL ICD on Jetson is selected when present.
pub struct HeadlessContext {
    pub gl: glow::Context,
    // Fields kept alive for the lifetime of the context (order matters for drop).
    _ctx: PossiblyCurrentContext,
    _surface: Surface<PbufferSurface>,
    _display: Display,
}

impl HeadlessContext {
    pub fn new() -> Result<Self> {
        // 1. Enumerate EGL devices and prefer the NVIDIA one.
        let devices: Vec<Device> = Device::query_devices()
            .context("EGL device enumeration failed — is libEGL installed?")?
            .collect();

        if devices.is_empty() {
            anyhow::bail!("No EGL devices found");
        }

        let device = devices
            .iter()
            .find(|d| {
                d.name()
                    .map(|n| n.contains("NVIDIA"))
                    .unwrap_or(false)
            })
            .unwrap_or(&devices[0]);

        tracing::info!(
            "EGL: using device {:?}",
            device.name().as_deref().unwrap_or("(unknown)")
        );

        // 2. Create a display from the device (no window handle = truly headless).
        let display = unsafe { Display::with_device(device, None) }
            .context("EGL display creation failed")?;

        // 3. Find a config that supports PBUFFER surfaces and 8-bit RGBA.
        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_surface_type(glutin::config::ConfigSurfaceTypes::PBUFFER)
            .build();

        let config = unsafe { display.find_configs(template) }
            .context("EGL find_configs failed")?
            .next()
            .context("No EGL PBUFFER config found")?;

        // 4. Create an OpenGL ES 3.0 context.
        let ctx_attrs = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(Some(Version::new(3, 0))))
            .build(None);

        let ctx = unsafe { display.create_context(&config, &ctx_attrs) }
            .context("EGL context creation failed")?;

        // 5. Create a 1×1 Pbuffer surface (only used to satisfy make_current).
        let surf_attrs = SurfaceAttributesBuilder::<PbufferSurface>::new()
            .build(NonZeroU32::new(1).unwrap(), NonZeroU32::new(1).unwrap());

        let surface = unsafe { display.create_pbuffer_surface(&config, &surf_attrs) }
            .context("EGL pbuffer surface creation failed")?;

        // 6. Make the context current on this thread.
        let ctx = ctx
            .make_current(&surface)
            .context("EGL make_current failed")?;

        // 7. Load glow function pointers via EGL getProcAddress.
        let gl = unsafe {
            glow::Context::from_loader_function(|sym| {
                let cstr = std::ffi::CString::new(sym).unwrap();
                display.get_proc_address(cstr.as_c_str()) as *const _
            })
        };

        tracing::info!("EGL headless context created (OpenGL ES 3.0)");

        Ok(Self {
            gl,
            _ctx: ctx,
            _surface: surface,
            _display: display,
        })
    }
}
