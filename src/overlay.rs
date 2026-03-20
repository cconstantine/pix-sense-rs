use crate::face::FaceDetection;
use egui::{Color32, Painter, Pos2, Stroke, StrokeKind};

const BBOX_COLOR: Color32 = Color32::from_rgb(0, 255, 255);
const EYE_COLOR: Color32 = Color32::from_rgb(0, 255, 0);
const LANDMARK_COLOR: Color32 = Color32::from_rgb(255, 255, 0);
const EYE_RADIUS: f32 = 6.0;
const LANDMARK_RADIUS: f32 = 3.0;

/// Landmark indices: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth

/// Draw face detections with eye landmarks on the egui painter.
pub fn draw_faces(
    painter: &Painter,
    faces: &[FaceDetection],
    offset: Pos2,
    scale_x: f32,
    scale_y: f32,
) {
    for face in faces {
        // Draw bounding box
        let rect = egui::Rect::from_min_max(
            Pos2::new(
                offset.x + face.bbox[0] * scale_x,
                offset.y + face.bbox[1] * scale_y,
            ),
            Pos2::new(
                offset.x + face.bbox[2] * scale_x,
                offset.y + face.bbox[3] * scale_y,
            ),
        );
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(1.5, BBOX_COLOR),
            StrokeKind::Outside,
        );

        // Draw confidence label
        let label = format!("{:.0}%", face.confidence * 100.0);
        painter.text(
            rect.left_top() + egui::vec2(2.0, -14.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::proportional(12.0),
            BBOX_COLOR,
        );

        // Draw landmarks
        for (i, lm) in face.landmarks.iter().enumerate() {
            let center = Pos2::new(
                offset.x + lm.x * scale_x,
                offset.y + lm.y * scale_y,
            );

            // Eyes get larger, green circles; other landmarks are smaller, yellow
            let (radius, color) = if i <= 1 {
                (EYE_RADIUS, EYE_COLOR)
            } else {
                (LANDMARK_RADIUS, LANDMARK_COLOR)
            };

            painter.circle_filled(center, radius, color);
            painter.circle_stroke(center, radius, Stroke::new(1.0, Color32::BLACK));
        }
    }
}
