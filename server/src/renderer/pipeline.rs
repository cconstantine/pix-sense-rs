use anyhow::{Context as _, Result};
use glow::HasContext as _;
use nalgebra::{Matrix4, Point3, Vector3};
use std::f32::consts::FRAC_PI_2;
use std::time::Instant;

use super::shaders;

/// Frame target: ~30 fps.
pub const FRAME_DURATION: std::time::Duration = std::time::Duration::from_millis(33);

/// Strip lines from user GLSL that conflict with the prefix we prepend
/// (version directives, precision qualifiers, and uniforms/outputs we already declare).
fn sanitize_user_glsl(user_glsl: &str) -> String {
    user_glsl
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("#version")
                && !trimmed.starts_with("precision ")
                && !trimmed.starts_with("uniform float time")
                && !trimmed.starts_with("uniform vec2 resolution")
                && !trimmed.starts_with("uniform vec2  resolution")
                && !trimmed.starts_with("uniform vec3 location")
                && !trimmed.starts_with("uniform vec3  location")
                && !trimmed.starts_with("out vec4 fragColor")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_pattern_frag_src(user_glsl: &str) -> String {
    format!("{}{}", shaders::PATTERN_FRAG_PREFIX, sanitize_user_glsl(user_glsl))
}

/// Pattern texture dimensions.
const PATTERN_SIZE: u32 = 512;

/// Fallback eye position when no person is tracked (metres above LED centroid).
const FALLBACK_EYE_OFFSET: f32 = 2.0;

// ---------------------------------------------------------------------------
// Helper: compile + link a GLSL program
// ---------------------------------------------------------------------------

pub fn compile_program(
    gl: &glow::Context,
    vert_src: &str,
    frag_src: &str,
) -> Result<glow::Program> {
    unsafe {
        let program = gl.create_program().map_err(|e| anyhow::anyhow!("{e}"))?;

        let vs = compile_shader(gl, glow::VERTEX_SHADER, vert_src)?;
        let fs = compile_shader(gl, glow::FRAGMENT_SHADER, frag_src)?;

        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);

        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            anyhow::bail!("Shader link error: {log}");
        }
        Ok(program)
    }
}

fn compile_shader(
    gl: &glow::Context,
    kind: u32,
    src: &str,
) -> Result<glow::Shader> {
    unsafe {
        let shader = gl.create_shader(kind).map_err(|e| anyhow::anyhow!("{e}"))?;
        gl.shader_source(shader, src);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            anyhow::bail!("Shader compile error: {log}");
        }
        Ok(shader)
    }
}

// ---------------------------------------------------------------------------
// Canvas size calculation
// ---------------------------------------------------------------------------

/// Compute a square power-of-two canvas big enough to hold `num_leds` pixels.
/// E.g. 1024 → (32, 32),  1500 → (64, 64).
pub fn compute_canvas_size(num_leds: usize) -> (u32, u32) {
    if num_leds == 0 {
        return (1, 1);
    }
    let side = (num_leds as f32).sqrt().ceil() as u32;
    let pot = side.next_power_of_two();
    (pot, pot)
}

// ---------------------------------------------------------------------------
// Main pipeline struct
// ---------------------------------------------------------------------------

pub struct RendererPipeline {
    // Pass 1 — pattern rendering
    pattern_fbo:      glow::Framebuffer,
    pattern_tex:      glow::Texture,
    pattern_program:  glow::Program,

    // Pass 2 — LED projection
    led_vao:          glow::VertexArray,
    #[allow(dead_code)] // held to keep VBO alive on the GPU
    led_vbo:          glow::Buffer,
    led_output_fbo:   glow::Framebuffer,
    #[allow(dead_code)] // held to keep texture alive; attached to FBO
    led_output_tex:   glow::Texture,
    led_proj_program: glow::Program,

    // Double-buffered PBOs for async GPU→CPU readback
    pbos:             [glow::Buffer; 2],
    pbo_frame:        usize,
    pbo_size:         usize,  // bytes: canvas_w * canvas_h * 4

    // Metadata
    pub num_leds:     usize,
    canvas_w:         u32,
    canvas_h:         u32,
    led_centroid:     [f32; 3],
    led_scope:        [f32; 3],
    start_time:       Instant,
}

impl RendererPipeline {
    /// Create all GL objects.
    /// `leds_by_device` lists LED positions per FadeCandy (already flat-mapped if needed).
    /// `all_leds` is the flat list used for VBO upload.
    pub fn new(
        gl: &glow::Context,
        all_leds: &[[f32; 3]],
        user_glsl: &str,
    ) -> Result<Self> {
        let num_leds = all_leds.len();
        let (canvas_w, canvas_h) = compute_canvas_size(num_leds);
        let pbo_size = (canvas_w * canvas_h * 4) as usize;

        // Compute LED centroid and bounding half-extents for view/projection construction.
        let (led_centroid, led_scope) = if num_leds > 0 {
            let (min, max) = all_leds.iter().fold(
                ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
                |(mn, mx), p| {
                    (
                        [mn[0].min(p[0]), mn[1].min(p[1]), mn[2].min(p[2])],
                        [mx[0].max(p[0]), mx[1].max(p[1]), mx[2].max(p[2])],
                    )
                },
            );
            let centroid = [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ];
            let scope = [
                (max[0] - min[0]) * 0.5,
                (max[1] - min[1]) * 0.5,
                (max[2] - min[2]) * 0.5,
            ];
            (centroid, scope)
        } else {
            ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
        };

        unsafe {
            // ------------------------------------------------------------------
            // Pass 1: pattern FBO + texture
            // ------------------------------------------------------------------
            let pattern_tex = gl.create_texture().map_err(|e| anyhow::anyhow!("{e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(pattern_tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                PATTERN_SIZE as i32,
                PATTERN_SIZE as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                None,
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.bind_texture(glow::TEXTURE_2D, None);

            let pattern_fbo = gl.create_framebuffer().map_err(|e| anyhow::anyhow!("{e}"))?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(pattern_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(pattern_tex),
                0,
            );
            check_fbo(gl).context("pattern FBO incomplete")?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            // ------------------------------------------------------------------
            // Pass 1: pattern shader program
            // ------------------------------------------------------------------
            let frag_src = build_pattern_frag_src(user_glsl);
            let pattern_program =
                compile_program(gl, shaders::PATTERN_VERT, &frag_src)
                    .context("pattern shader compile failed")?;

            // ------------------------------------------------------------------
            // Pass 2: LED VBO + VAO
            // ------------------------------------------------------------------
            let led_vao = gl.create_vertex_array().map_err(|e| anyhow::anyhow!("{e}"))?;
            let led_vbo = gl.create_buffer().map_err(|e| anyhow::anyhow!("{e}"))?;

            gl.bind_vertex_array(Some(led_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(led_vbo));

            // Layout: [x:f32, y:f32, z:f32, idx:f32] — 16 bytes per LED vertex.
            let vbo_data: Vec<f32> = all_leds
                .iter()
                .enumerate()
                .flat_map(|(i, p)| [p[0], p[1], p[2], i as f32])
                .collect();
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&vbo_data),
                glow::STATIC_DRAW,
            );

            // attribute 0: vec3 led_world_pos
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 16, 0);
            // attribute 1: float led_index
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 1, glow::FLOAT, false, 16, 12);

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);

            // ------------------------------------------------------------------
            // Pass 2: LED output FBO + texture
            // ------------------------------------------------------------------
            let led_output_tex = gl.create_texture().map_err(|e| anyhow::anyhow!("{e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(led_output_tex));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA8 as i32,
                canvas_w as i32,
                canvas_h as i32,
                0,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                None,
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.bind_texture(glow::TEXTURE_2D, None);

            let led_output_fbo = gl.create_framebuffer().map_err(|e| anyhow::anyhow!("{e}"))?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(led_output_fbo));
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(led_output_tex),
                0,
            );
            check_fbo(gl).context("led_output FBO incomplete")?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            // ------------------------------------------------------------------
            // Pass 2: LED projection shader
            // ------------------------------------------------------------------
            let led_proj_program =
                compile_program(gl, shaders::LED_PROJ_VERT, shaders::LED_PROJ_FRAG)
                    .context("LED projection shader compile failed")?;

            // ------------------------------------------------------------------
            // Double-buffered PBOs
            // ------------------------------------------------------------------
            let pbo0 = gl.create_buffer().map_err(|e| anyhow::anyhow!("{e}"))?;
            let pbo1 = gl.create_buffer().map_err(|e| anyhow::anyhow!("{e}"))?;
            for pbo in [pbo0, pbo1] {
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
                gl.buffer_data_size(
                    glow::PIXEL_PACK_BUFFER,
                    pbo_size as i32,
                    glow::STREAM_READ,
                );
            }
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);

            Ok(Self {
                pattern_fbo,
                pattern_tex,
                pattern_program,
                led_vao,
                led_vbo,
                led_output_fbo,
                led_output_tex,
                led_proj_program,
                pbos: [pbo0, pbo1],
                pbo_frame: 0,
                pbo_size,
                num_leds,
                canvas_w,
                canvas_h,
                led_centroid,
                led_scope,
                start_time: Instant::now(),
            })
        }
    }

    // -----------------------------------------------------------------------
    // Pass 1: render GLSL pattern to 512×512 texture
    // -----------------------------------------------------------------------
    pub fn render_pattern(&self, gl: &glow::Context, location: [f32; 3]) {
        let elapsed_secs = self.start_time.elapsed().as_secs_f32();

        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.pattern_fbo));
            gl.viewport(0, 0, PATTERN_SIZE as i32, PATTERN_SIZE as i32);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(self.pattern_program));

            if let Some(loc) = gl.get_uniform_location(self.pattern_program, "time") {
                gl.uniform_1_f32(Some(&loc), elapsed_secs);
            }
            if let Some(loc) = gl.get_uniform_location(self.pattern_program, "resolution") {
                gl.uniform_2_f32(Some(&loc), PATTERN_SIZE as f32, PATTERN_SIZE as f32);
            }
            if let Some(loc) = gl.get_uniform_location(self.pattern_program, "location") {
                gl.uniform_3_f32(Some(&loc), location[0], location[1], location[2]);
            }

            // Full-screen triangle — no VBO (vertex shader uses gl_VertexID).
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.use_program(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    // -----------------------------------------------------------------------
    // Pass 2: project LED positions onto pattern texture, output one px/LED
    // -----------------------------------------------------------------------
    pub fn render_leds(
        &self,
        gl: &glow::Context,
        person_pos: Option<[f32; 3]>,
        overscan: bool,
    ) {
        if self.num_leds == 0 {
            return;
        }

        let vp = build_view_proj(person_pos, self.led_centroid, self.led_scope, overscan);

        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.led_output_fbo));
            gl.viewport(0, 0, self.canvas_w as i32, self.canvas_h as i32);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.use_program(Some(self.led_proj_program));

            // Bind pattern texture to unit 0
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.pattern_tex));
            if let Some(loc) = gl.get_uniform_location(self.led_proj_program, "pattern_tex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }

            if let Some(loc) = gl.get_uniform_location(self.led_proj_program, "view_proj") {
                gl.uniform_matrix_4_f32_slice(Some(&loc), false, &vp);
            }
            if let Some(loc) = gl.get_uniform_location(self.led_proj_program, "canvas_size") {
                gl.uniform_2_f32(Some(&loc), self.canvas_w as f32, self.canvas_h as f32);
            }

            gl.bind_vertex_array(Some(self.led_vao));
            gl.draw_arrays(glow::POINTS, 0, self.num_leds as i32);
            gl.bind_vertex_array(None);

            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.use_program(None);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    // -----------------------------------------------------------------------
    // PBO double-buffered readback
    // -----------------------------------------------------------------------
    /// Trigger async readback into PBO[frame % 2], then map and return the
    /// *previous* frame's data as raw RGBA bytes.  Returns `None` on the very
    /// first frame (no previous data) or if mapping fails.
    pub fn pbo_readback(&mut self, gl: &glow::Context) -> Option<Vec<u8>> {
        let write_idx = self.pbo_frame % 2;
        let read_idx = 1 - write_idx;

        unsafe {
            // --- trigger async readback for this frame ---
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(self.led_output_fbo));
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(self.pbos[write_idx]));
            gl.read_pixels(
                0,
                0,
                self.canvas_w as i32,
                self.canvas_h as i32,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::BufferOffset(0),
            );
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);

            // --- read the previous frame's PBO ---
            if self.pbo_frame == 0 {
                // No previous frame yet.
                self.pbo_frame += 1;
                return None;
            }

            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(self.pbos[read_idx]));
            let ptr = gl.map_buffer_range(
                glow::PIXEL_PACK_BUFFER,
                0,
                self.pbo_size as i32,
                glow::MAP_READ_BIT,
            );

            let result = if ptr.is_null() {
                tracing::warn!("PBO map_buffer_range returned null");
                None
            } else {
                let slice = std::slice::from_raw_parts(ptr as *const u8, self.pbo_size);
                Some(slice.to_vec())
            };

            gl.unmap_buffer(glow::PIXEL_PACK_BUFFER);
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);

            self.pbo_frame += 1;
            result
        }
    }

    // -----------------------------------------------------------------------
    // Shader hot-reload (called when pattern changes in DB)
    // -----------------------------------------------------------------------
    pub fn reload_pattern_shader(&mut self, gl: &glow::Context, user_glsl: &str) {
        let frag_src = build_pattern_frag_src(user_glsl);
        match compile_program(gl, shaders::PATTERN_VERT, &frag_src) {
            Ok(prog) => {
                unsafe { gl.delete_program(self.pattern_program) };
                self.pattern_program = prog;
                self.start_time = Instant::now(); // reset time uniform
                tracing::info!("Pattern shader reloaded successfully");
            }
            Err(e) => {
                tracing::error!("Pattern shader reload failed (keeping old shader): {e:#}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Extract per-LED RGB values from RGBA canvas data, applying brightness/gamma
    // -----------------------------------------------------------------------
    pub fn extract_led_rgb(
        rgba: &[u8],
        num_leds: usize,
        brightness: f32,
        gamma: f32,
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(num_leds * 3);
        for i in 0..num_leds {
            let base = i * 4;
            if base + 3 >= rgba.len() {
                out.extend_from_slice(&[0, 0, 0]);
                continue;
            }
            let r = apply_brightness_gamma(rgba[base],     brightness, gamma);
            let g = apply_brightness_gamma(rgba[base + 1], brightness, gamma);
            let b = apply_brightness_gamma(rgba[base + 2], brightness, gamma);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn apply_brightness_gamma(v: u8, brightness: f32, gamma: f32) -> u8 {
    let f = (v as f32 / 255.0).powf(gamma) * brightness;
    (f.clamp(0.0, 1.0) * 255.0) as u8
}

/// Build a perspective view-projection matrix from `person_pos` looking toward
/// the LED centroid.  Returns a column-major `[f32; 16]` for OpenGL.
///
/// FOV is picked from the LED bounding-box half-extents `scope`:
///   overscan = true  → frustum contains the LED bounding sphere (every LED
///                      lands inside the pattern, minimal wasted pattern).
///   overscan = false → frustum fits inside the LED cloud (every pixel of the
///                      pattern lands on at least one LED; outer LEDs go black).
/// Mirrors pixo's `IsoCamera::get_zoom` (pixlib/src/camera.cpp).
fn build_view_proj(
    person_pos: Option<[f32; 3]>,
    centroid: [f32; 3],
    scope: [f32; 3],
    overscan: bool,
) -> [f32; 16] {
    let eye = match person_pos {
        Some(p) => Point3::new(p[0], p[1], p[2]),
        None => Point3::new(
            centroid[0],
            centroid[1] + FALLBACK_EYE_OFFSET,
            centroid[2],
        ),
    };

    let target = Point3::new(centroid[0], centroid[1], centroid[2]);

    // Avoid degenerate view matrix when eye == target.
    let look_dir = target - eye;
    let dist = look_dir.norm();
    if dist < 1e-6 {
        return Matrix4::identity().as_slice().try_into().unwrap();
    }

    // Choose up vector; if looking straight up/down, use Z instead.
    let up = if look_dir.y.abs() > 0.99 * dist {
        Vector3::z()
    } else {
        Vector3::y()
    };

    let fov = compute_fov(scope, dist, overscan);

    let view = Matrix4::look_at_rh(&eye, &target, &up);
    let proj = Matrix4::new_perspective(1.0, fov, 0.01, 100.0);
    let vp = proj * view;

    // nalgebra stores column-major — convert to [f32; 16].
    let s = vp.as_slice();
    s.try_into().unwrap()
}

/// Vertical FOV (radians) for the LED-projection perspective.  Ports pixo's
/// `IsoCamera::get_zoom`.  `scope` is the LED-cloud half-extents; `dist` is the
/// eye→centroid distance.  Falls back to 90° when the geometry is degenerate
/// (e.g. viewpoint inside the LED cloud).
fn compute_fov(scope: [f32; 3], dist: f32, overscan: bool) -> f32 {
    const MIN_FOV: f32 = 10.0 * std::f32::consts::PI / 180.0;
    const MAX_FOV: f32 = 170.0 * std::f32::consts::PI / 180.0;

    let fov = if overscan {
        let radius = (scope[0] * scope[0] + scope[1] * scope[1] + scope[2] * scope[2]).sqrt();
        if dist <= 1e-6 {
            return FRAC_PI_2;
        }
        2.0 * (radius / dist).atan()
    } else {
        let edge = 2.0 * (scope[0] + scope[1] + scope[2]) / 3.0;
        let position_distance = dist - edge / 3.0;
        if position_distance <= 1e-6 {
            return FRAC_PI_2;
        }
        2.0 * ((edge * 0.5) / position_distance).atan()
    };

    fov.clamp(MIN_FOV, MAX_FOV)
}

fn check_fbo(gl: &glow::Context) -> Result<()> {
    unsafe {
        let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
        if status == glow::FRAMEBUFFER_COMPLETE {
            Ok(())
        } else {
            anyhow::bail!("Framebuffer status: 0x{:x}", status)
        }
    }
}
