/// Vertex shader for Pass 1 (pattern rendering).
/// Generates a full-screen triangle using gl_VertexID — no VBO needed.
pub const PATTERN_VERT: &str = r#"#version 300 es
// Full-screen triangle: vertices are computed from gl_VertexID so no VBO is needed.
const vec2 POSITIONS[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);
void main() {
    gl_Position = vec4(POSITIONS[gl_VertexID], 0.0, 1.0);
}
"#;

/// Prefix injected before user GLSL code for Pass 1 fragment shader.
/// The user's code must write to `fragColor`.
pub const PATTERN_FRAG_PREFIX: &str = r#"#version 300 es
precision highp float;
uniform float time;       // seconds since renderer start
uniform vec2  resolution; // texture dimensions (512.0, 512.0)
uniform vec3  location;   // tracked person world position (metres)
out vec4 fragColor;
"#;

/// Vertex shader for Pass 2 (LED projection).
/// Each vertex represents one LED.  The shader:
///   1. Projects the LED's world position through `view_proj` (person's viewpoint).
///   2. Maps clip coords → UV, samples the pattern texture.
///   3. Computes pixo-style distance compensation `(led_r / cam_r)^2`.
///   4. Positions the output GL_POINT at the LED's slot in the output canvas.
pub const LED_PROJ_VERT: &str = r#"#version 300 es
layout(location = 0) in vec3 led_world_pos;
layout(location = 1) in float led_index;

uniform mat4      view_proj;     // perspective from person's viewpoint
uniform sampler2D pattern_tex;   // 512x512 rendered pattern
uniform vec2      canvas_size;   // output canvas dimensions (e.g. vec2(32.0, 32.0))
uniform vec3      camera_pos;    // tracked person world position
uniform float     inv_cam_r_sq;  // 1.0 / |camera_pos|^2, precomputed CPU-side
uniform float     comp_enabled;  // 1.0 when tracking + origin normalisation valid, else 0.0

out vec4  led_color;
out float distance_compensation;

void main() {
    // Project LED world position to clip space via the person's view frustum.
    vec4 clip = view_proj * vec4(led_world_pos, 1.0);

    // Map NDC [-1,1] → UV [0,1].
    vec2 uv = clip.xy / clip.w * 0.5 + 0.5;

    // Only LEDs inside the pattern frustum sample the pattern; those outside
    // go black.  In overscan mode the frustum contains every LED (all sample);
    // in standard mode the outermost LEDs fall outside and remain dark.
    bool inside = clip.w > 0.0
               && all(greaterThanEqual(uv, vec2(0.0)))
               && all(lessThanEqual   (uv, vec2(1.0)));
    led_color = inside ? texture(pattern_tex, uv) : vec4(0.0, 0.0, 0.0, 1.0);

    // Per-LED distance compensation: (led_r / cam_r)^2 (pixo's led_mesh shader).
    // `cam_r` is |camera_pos|, i.e. the world origin is the reference point.
    vec3  d        = camera_pos - led_world_pos;
    float led_r_sq = dot(d, d);
    distance_compensation = mix(1.0, led_r_sq * inv_cam_r_sq, comp_enabled);

    // Compute the pixel column and row for this LED in the output canvas.
    float col = mod(led_index, canvas_size.x);
    float row = floor(led_index / canvas_size.x);

    // Convert canvas pixel to NDC of the output FBO.
    float x = (col + 0.5) / canvas_size.x * 2.0 - 1.0;
    float y = (row + 0.5) / canvas_size.y * 2.0 - 1.0;

    gl_Position  = vec4(x, y, 0.0, 1.0);
    gl_PointSize = 1.0;
}
"#;

/// Fragment shader for Pass 2.  Applies gamma, brightness, and the distance
/// compensation computed per-vertex, matching pixo's fragment shader.
pub const LED_PROJ_FRAG: &str = r#"#version 300 es
precision highp float;
in  vec4  led_color;
in  float distance_compensation;

uniform float brightness;
uniform float gamma;

out vec4 fragColor;
void main() {
    vec3 c = pow(led_color.rgb, vec3(gamma)) * brightness * distance_compensation;
    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"#;
