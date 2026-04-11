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
    Both,
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
            stream: StreamSelection::Both,
        }
    }
}

/// Metadata sent alongside JPEG-encoded frames over WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMetadata {
    pub rgb_faces: Vec<FaceDetection>,
    pub ir_faces: Vec<FaceDetection>,
    pub rgb_size: [u32; 2],
    pub ir_size: [u32; 2],
    pub depth_size: [u32; 2],
    /// The algorithm + stream configuration that was active when this frame was processed.
    pub active_config: DetectionConfig,
}

/// Binary message format:
///   [u32 LE: rgb_jpeg_len][rgb_jpeg bytes]
///   [u32 LE: ir_jpeg_len][ir_jpeg bytes]
///   [u32 LE: depth_jpeg_len][depth_jpeg bytes]
///   [remaining bytes: JSON of FrameMetadata]

pub fn encode_frame_message(
    rgb_jpeg: &[u8],
    ir_jpeg: &[u8],
    depth_jpeg: &[u8],
    metadata: &FrameMetadata,
) -> Vec<u8> {
    let json = serde_json::to_vec(metadata).expect("failed to serialize metadata");
    let mut msg =
        Vec::with_capacity(12 + rgb_jpeg.len() + ir_jpeg.len() + depth_jpeg.len() + json.len());
    msg.extend_from_slice(&(rgb_jpeg.len() as u32).to_le_bytes());
    msg.extend_from_slice(rgb_jpeg);
    msg.extend_from_slice(&(ir_jpeg.len() as u32).to_le_bytes());
    msg.extend_from_slice(ir_jpeg);
    msg.extend_from_slice(&(depth_jpeg.len() as u32).to_le_bytes());
    msg.extend_from_slice(depth_jpeg);
    msg.extend_from_slice(&json);
    msg
}

pub fn decode_frame_message(data: &[u8]) -> Option<(&[u8], &[u8], &[u8], FrameMetadata)> {
    if data.len() < 12 {
        return None;
    }

    let rgb_len = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    if data.len() < 4 + rgb_len + 4 {
        return None;
    }

    let rgb_jpeg = &data[4..4 + rgb_len];
    let rest = &data[4 + rgb_len..];

    let ir_len = u32::from_le_bytes(rest[0..4].try_into().ok()?) as usize;
    if rest.len() < 4 + ir_len + 4 {
        return None;
    }

    let ir_jpeg = &rest[4..4 + ir_len];
    let rest = &rest[4 + ir_len..];

    let depth_len = u32::from_le_bytes(rest[0..4].try_into().ok()?) as usize;
    if rest.len() < 4 + depth_len {
        return None;
    }

    let depth_jpeg = &rest[4..4 + depth_len];
    let json_bytes = &rest[4 + depth_len..];

    let metadata: FrameMetadata = serde_json::from_slice(json_bytes).ok()?;

    Some((rgb_jpeg, ir_jpeg, depth_jpeg, metadata))
}
