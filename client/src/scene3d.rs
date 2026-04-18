//! GL-backed 3D renderer for LED cubes with depth testing.
//!
//! Rendered inside egui via `egui_glow::CallbackFn`. Only LED geometry lives
//! here — 2D overlays (axes, frustum, tracking dots, labels) stay in the egui
//! painter and compose on top of this pass.
//!
//! The renderer deliberately holds only GL resource handles (which are `Copy`
//! slotmap keys on wasm, trivially `Send + Sync`) and never the `glow::Context`,
//! so it can live inside an `Arc<Mutex<…>>` used by egui's `Send + Sync`
//! paint-callback closure. The context is borrowed from the callback painter
//! on each paint.

use eframe::glow::{self, HasContext};

pub struct SceneRenderer {
    program: glow::Program,
    vao: glow::VertexArray,
    // Owned by the VAO's attribute bindings; we never read them back directly
    // but must keep them alive to avoid leaking (and for `destroy`).
    vbo_cube: glow::Buffer,
    ibo_cube: glow::Buffer,
    vbo_inst: glow::Buffer,
    inst_capacity_bytes: usize,
    u_mvp: glow::UniformLocation,
    u_half: glow::UniformLocation,
    u_brightness: glow::UniformLocation,
    u_gamma: glow::UniformLocation,
}

const VS_SRC: &str = r#"#version 300 es
precision highp float;
layout(location = 0) in vec3 a_pos;
layout(location = 2) in vec3 a_inst_pos;
layout(location = 3) in vec3 a_inst_color;
uniform mat4 u_mvp;
uniform float u_half_size;
// Server encodes each LED as `clamp(pow(orig, gamma) * brightness * dist_comp, 0, 1)`.
// We invert here so the scene shows `orig` regardless of slider position.
// Two regimes:
//   • Unsaturated channel: recv < 1, reverse via `pow(recv / brightness, 1/gamma)`.
//   • Saturated channel (recv ≈ 1): server clipped, orig was ≥ some value — best
//     guess is 1.0, so we pin it there to avoid the scene dimming as the slider
//     moves past 1.0 into the clipping regime.
// `distance_compensation` is per-LED view-dependent and ~1 in common use; ignored.
uniform float u_brightness;
uniform float u_gamma;
out vec3 v_color;
void main() {
    vec3 world = a_inst_pos + a_pos * u_half_size;
    gl_Position = u_mvp * vec4(world, 1.0);
    float b = max(u_brightness, 1e-3);
    float inv_g = 1.0 / max(u_gamma, 1e-3);
    vec3 inv = pow(a_inst_color / b, vec3(inv_g));
    // u8 quantization: 255/255 = 1.0, but 254/255 ≈ 0.996 — treat both as saturated.
    vec3 saturated = step(vec3(253.5 / 255.0), a_inst_color);
    v_color = clamp(mix(inv, vec3(1.0), saturated), 0.0, 1.0);
}
"#;

const FS_SRC: &str = r#"#version 300 es
precision highp float;
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"#;

// 8 shared corners at (±1, ±1, ±1). Corner index bit pattern: bit0=X, bit1=Y, bit2=Z.
#[rustfmt::skip]
const CUBE_VERTICES: [f32; 8 * 3] = [
    -1.0, -1.0, -1.0,  //  0 (-X,-Y,-Z)
     1.0, -1.0, -1.0,  //  1 (+X,-Y,-Z)
    -1.0,  1.0, -1.0,  //  2 (-X,+Y,-Z)
     1.0,  1.0, -1.0,  //  3 (+X,+Y,-Z)
    -1.0, -1.0,  1.0,  //  4 (-X,-Y,+Z)
     1.0, -1.0,  1.0,  //  5 (+X,-Y,+Z)
    -1.0,  1.0,  1.0,  //  6 (-X,+Y,+Z)
     1.0,  1.0,  1.0,  //  7 (+X,+Y,+Z)
];

// 12 triangles, CCW when viewed from outside so back-face culling works.
#[rustfmt::skip]
const CUBE_INDICES: [u16; 36] = [
    1, 3, 7,  1, 7, 5,   // +X
    0, 4, 6,  0, 6, 2,   // -X
    2, 6, 7,  2, 7, 3,   // +Y
    0, 1, 5,  0, 5, 4,   // -Y
    4, 5, 7,  4, 7, 6,   // +Z
    0, 2, 3,  0, 3, 1,   // -Z
];

impl SceneRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let vs = compile_shader(gl, glow::VERTEX_SHADER, VS_SRC)?;
            let fs = compile_shader(gl, glow::FRAGMENT_SHADER, FS_SRC)?;
            let program = gl
                .create_program()
                .map_err(|e| format!("create_program: {e}"))?;
            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                gl.delete_shader(vs);
                gl.delete_shader(fs);
                return Err(format!("link: {log}"));
            }
            gl.detach_shader(program, vs);
            gl.detach_shader(program, fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);

            let u_mvp = gl
                .get_uniform_location(program, "u_mvp")
                .ok_or("missing uniform u_mvp")?;
            let u_half = gl
                .get_uniform_location(program, "u_half_size")
                .ok_or("missing uniform u_half_size")?;
            let u_brightness = gl
                .get_uniform_location(program, "u_brightness")
                .ok_or("missing uniform u_brightness")?;
            let u_gamma = gl
                .get_uniform_location(program, "u_gamma")
                .ok_or("missing uniform u_gamma")?;

            let vao = gl
                .create_vertex_array()
                .map_err(|e| format!("vao: {e}"))?;
            gl.bind_vertex_array(Some(vao));

            let vbo_cube = gl.create_buffer().map_err(|e| format!("vbo_cube: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_cube));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck::cast_slice(&CUBE_VERTICES),
                glow::STATIC_DRAW,
            );
            // a_pos (loc 0): tightly packed vec3, stride 12 bytes
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.enable_vertex_attrib_array(0);

            let ibo_cube = gl.create_buffer().map_err(|e| format!("ibo_cube: {e}"))?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo_cube));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck::cast_slice(&CUBE_INDICES),
                glow::STATIC_DRAW,
            );

            let vbo_inst = gl.create_buffer().map_err(|e| format!("vbo_inst: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_inst));
            let initial_bytes = 64usize * 6 * 4;
            gl.buffer_data_size(
                glow::ARRAY_BUFFER,
                initial_bytes as i32,
                glow::DYNAMIC_DRAW,
            );
            // a_inst_pos (loc 2), a_inst_color (loc 3): stride 6*f32 = 24 bytes, divisor 1
            gl.vertex_attrib_pointer_f32(2, 3, glow::FLOAT, false, 24, 0);
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_divisor(2, 1);
            gl.vertex_attrib_pointer_f32(3, 3, glow::FLOAT, false, 24, 12);
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_divisor(3, 1);

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, None);

            Ok(SceneRenderer {
                program,
                vao,
                vbo_cube,
                ibo_cube,
                vbo_inst,
                inst_capacity_bytes: initial_bytes,
                u_mvp,
                u_half,
                u_brightness,
                u_gamma,
            })
        }
    }

    /// Render LED cubes. `instances` is `[x, y, z, r, g, b]` per LED.
    /// `viewport_px` is `(x, y, w, h)` in framebuffer pixels (GL origin bottom-left).
    pub fn paint(
        &mut self,
        gl: &glow::Context,
        viewport_px: (i32, i32, i32, i32),
        mvp: &[f32; 16],
        half_size_m: f32,
        brightness: f32,
        gamma: f32,
        instances: &[f32],
    ) {
        let count = instances.len() / 6;
        if count == 0 {
            return;
        }
        unsafe {
            gl.viewport(viewport_px.0, viewport_px.1, viewport_px.2, viewport_px.3);
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(viewport_px.0, viewport_px.1, viewport_px.2, viewport_px.3);
            gl.clear(glow::DEPTH_BUFFER_BIT);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LESS);
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::BACK);
            gl.front_face(glow::CCW);

            gl.use_program(Some(self.program));
            gl.uniform_matrix_4_f32_slice(Some(&self.u_mvp), false, mvp);
            gl.uniform_1_f32(Some(&self.u_half), half_size_m);
            gl.uniform_1_f32(Some(&self.u_brightness), brightness);
            gl.uniform_1_f32(Some(&self.u_gamma), gamma);

            gl.bind_vertex_array(Some(self.vao));

            let bytes: &[u8] = bytemuck::cast_slice(instances);
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo_inst));
            if bytes.len() > self.inst_capacity_bytes {
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::DYNAMIC_DRAW);
                self.inst_capacity_bytes = bytes.len();
            } else {
                gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
            }

            gl.draw_elements_instanced(
                glow::TRIANGLES,
                CUBE_INDICES.len() as i32,
                glow::UNSIGNED_SHORT,
                0,
                count as i32,
            );

            // Leave GL state clean enough for egui's next draw.
            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::SCISSOR_TEST);
        }
    }

    /// Call from `eframe::App::on_exit(gl)` to free GPU resources.
    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.vbo_cube);
            gl.delete_buffer(self.ibo_cube);
            gl.delete_buffer(self.vbo_inst);
        }
    }
}

unsafe fn compile_shader(
    gl: &glow::Context,
    kind: u32,
    src: &str,
) -> Result<glow::Shader, String> {
    let sh = gl
        .create_shader(kind)
        .map_err(|e| format!("create_shader: {e}"))?;
    gl.shader_source(sh, src);
    gl.compile_shader(sh);
    if !gl.get_shader_compile_status(sh) {
        let log = gl.get_shader_info_log(sh);
        gl.delete_shader(sh);
        return Err(format!("compile: {log}"));
    }
    Ok(sh)
}

// Column-major 4×4 matrix helpers.

pub fn perspective(fovy_rad: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fovy_rad * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) * nf, -1.0,
        0.0, 0.0, 2.0 * far * near * nf, 0.0,
    ]
}

pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize(sub(target, eye));
    let r = normalize(cross(f, up));
    let u = cross(r, f);
    [
        r[0],  u[0], -f[0], 0.0,
        r[1],  u[1], -f[1], 0.0,
        r[2],  u[2], -f[2], 0.0,
        -dot(r, eye), -dot(u, eye), dot(f, eye), 1.0,
    ]
}

pub fn mul_mat4(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for c in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k * 4 + row] * b[c * 4 + k];
            }
            r[c * 4 + row] = s;
        }
    }
    r
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}
