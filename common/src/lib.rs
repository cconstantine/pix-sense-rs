use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FaceLandmark {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetection {
    pub bbox: [f32; 4], // x1, y1, x2, y2
    pub confidence: f32,
    /// 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    /// None when detected from behind or when only head (not face) was found.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub landmarks: Option<[FaceLandmark; 5]>,
    /// 3D position in camera frame (metres): [X right, Y down, Z forward].
    /// None when depth data is unavailable at the face centre.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xyz: Option<[f32; 3]>,
}

/// Which detection algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionAlgo {
    /// Fast YOLOv8n head detector — all orientations, no landmarks (~5-10 ms GPU).
    YoloHead,
    /// SCRFD face detector on the full frame — frontal/angled with 5-point landmarks (~10-15 ms GPU).
    ScrfdFace,
    /// Two-stage: YOLO finds head ROIs, SCRFD refines with landmarks (~15-25 ms GPU).
    YoloHeadScrfdLandmarks,
}

/// Which camera stream(s) to run detection on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamSelection {
    Rgb,
    Ir,
}

/// Runtime detection configuration — shared between server and client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub algo: DetectionAlgo,
    pub stream: StreamSelection,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            algo: DetectionAlgo::YoloHead,
            stream: StreamSelection::Rgb,
        }
    }
}

/// Metadata sent alongside JPEG-encoded frames over WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMetadata {
    /// RealSense serial of the camera this frame came from. Clients demux by this.
    pub camera_id: String,
    pub rgb_faces: Vec<FaceDetection>,
    pub ir_faces: Vec<FaceDetection>,
    pub rgb_size: [u32; 2],
    pub ir_size: [u32; 2],
    /// The algorithm + stream configuration that was active when this frame was processed.
    pub active_config: DetectionConfig,
    /// ROI crop rectangle [x1, y1, x2, y2] in full-image pixel coordinates.
    #[serde(default)]
    pub roi_rect: Option<[u32; 4]>,
    /// Index into `rgb_faces` of the person currently being tracked (if any).
    #[serde(default)]
    pub tracked_rgb_idx: Option<usize>,
    /// Index into `ir_faces` of the person currently being tracked (if any).
    #[serde(default)]
    pub tracked_ir_idx: Option<usize>,
}

/// Binary message format:
///   [u32 LE: rgb_jpeg_len][rgb_jpeg bytes]
///   [u32 LE: ir_jpeg_len][ir_jpeg bytes]
///   [remaining bytes: JSON of FrameMetadata]

/// A single LED physical position in world-frame metres — pixo convention
/// (X=right, Y=up, Z=forward, right-handed OpenGL).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LedPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A live tracking location from the `tracking_locations` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackingPoint {
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Tagged text message sent from server to WebSocket clients.
/// Serialises as `{"type":"tracking","data":[...]}` etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ServerMessage {
    /// Live person tracking locations.
    Tracking(Vec<TrackingPoint>),
    /// Current RGB color for each LED, in the same order as `/api/leds`.
    LedColors(Vec<[u8; 3]>),
    /// Active detection config — sent to new clients on connect so the UI
    /// reflects the persisted server-side setting immediately.
    Config(DetectionConfig),
}

/// Declares which view the client is currently rendering. The server uses
/// this to skip work for data the client won't display — camera frames while
/// the user is in Scene, LED colors / tracking while the user is in Cameras.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ViewSubscription {
    /// Only the named camera's frames. No tracking / LedColors.
    Camera { camera_id: String },
    /// Tracking + LedColors. No camera frames.
    Scene,
}

impl Default for ViewSubscription {
    fn default() -> Self {
        Self::Scene
    }
}

/// Tagged text message sent from WebSocket clients to the server.
/// Serialises as `{"type":"config","data":{...}}` or
/// `{"type":"select_person","data":{"xyz":[x,y,z]}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Update the active detection algorithm / stream.
    Config(DetectionConfig),
    /// Lock person tracking onto the person at this camera-frame XYZ.
    /// `camera_id` selects which camera's extrinsics to apply so the server can
    /// convert to the world-frame target.
    SelectPerson { camera_id: String, xyz: [f32; 3] },
    /// Drive the LED renderer from a simulated world-frame location.
    /// `Some(xyz)` sets/updates the override; `None` clears it and lets the
    /// real DB-driven tracking resume.
    VirtualLocation { xyz: Option<[f32; 3]> },
    /// Enable or disable tracking updates to the renderer. When `false`, the
    /// server stops pushing new positions (from DB or virtual) to the renderer's
    /// watch channel, so the LEDs hold whatever state was last sent.
    TrackingEnabled(bool),
    /// Client is now on this view — server skips pushes that aren't needed.
    View(ViewSubscription),
}

/// Camera extrinsic transform: p_world = R * p_cam + t
///
/// Camera frame is the depth-sensor native OpenCV convention (X=right, Y=down, Z=forward).
/// World frame is the pixo convention (X=right, Y=up, Z=forward, right-handed OpenGL).
/// For a camera mounted upright with no roll, R is a 180° rotation about X =
/// `diag(1, -1, -1)`, which is what bridges the two right-handed frames.
/// R is a row-major 3×3 rotation matrix. The camera's origin in world coordinates is `t`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CameraExtrinsics {
    /// Row-major 3×3 rotation matrix: r[row][col]
    pub r: [[f32; 3]; 3],
    /// Translation vector (metres, world frame). Camera origin in world = t.
    pub t: [f32; 3],
}

impl CameraExtrinsics {
    /// Transform a camera-frame point to a world-frame point.
    #[inline]
    pub fn apply(&self, p: [f32; 3]) -> [f32; 3] {
        let r = &self.r;
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + self.t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + self.t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + self.t[2],
        ]
    }
}

/// A single point correspondence used for extrinsics calibration.
/// `cam` is in camera frame (from the depth camera); `world` is the same physical
/// point measured in the external reference frame.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CalibrationPoint {
    pub cam: [f32; 3],
    pub world: [f32; 3],
}

/// A GLSL shader pattern for LED visualisation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Pattern {
    pub name: String,
    pub glsl_code: String,
    pub enabled: bool,
    pub overscan: bool,
}

/// Fields that can be updated on an existing pattern (name is immutable after creation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternUpdate {
    pub glsl_code: String,
    pub enabled: bool,
    pub overscan: bool,
}

/// Per-sculpture display settings applied to the rendered LED output.
/// `brightness` is a linear multiplier (1.0 = unchanged); `gamma` is the
/// decoding exponent applied before the multiply (2.2 for sRGB-ish output).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SculptureSettings {
    pub brightness: f32,
    pub gamma: f32,
    /// Interval between random auto-rotations of the active pattern, in minutes.
    /// `0` disables rotation.
    #[serde(default)]
    pub rotation_minutes: f32,
}

pub fn encode_frame_message(
    rgb_jpeg: &[u8],
    ir_jpeg: &[u8],
    metadata: &FrameMetadata,
) -> Vec<u8> {
    let json = serde_json::to_vec(metadata).expect("failed to serialize metadata");
    let mut msg =
        Vec::with_capacity(8 + rgb_jpeg.len() + ir_jpeg.len() + json.len());
    msg.extend_from_slice(&(rgb_jpeg.len() as u32).to_le_bytes());
    msg.extend_from_slice(rgb_jpeg);
    msg.extend_from_slice(&(ir_jpeg.len() as u32).to_le_bytes());
    msg.extend_from_slice(ir_jpeg);
    msg.extend_from_slice(&json);
    msg
}

pub fn decode_frame_message(data: &[u8]) -> Option<(&[u8], &[u8], FrameMetadata)> {
    if data.len() < 8 {
        return None;
    }

    let rgb_len = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    if data.len() < 4 + rgb_len + 4 {
        return None;
    }

    let rgb_jpeg = &data[4..4 + rgb_len];
    let rest = &data[4 + rgb_len..];

    let ir_len = u32::from_le_bytes(rest[0..4].try_into().ok()?) as usize;
    if rest.len() < 4 + ir_len {
        return None;
    }

    let ir_jpeg = &rest[4..4 + ir_len];
    let json_bytes = &rest[4 + ir_len..];

    let metadata: FrameMetadata = serde_json::from_slice(json_bytes).ok()?;

    Some((rgb_jpeg, ir_jpeg, metadata))
}
