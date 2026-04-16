use egui::{Color32, Painter, Pos2, Stroke, StrokeKind};
use pix_sense_common::FaceDetection;

const BBOX_COLOR: Color32 = Color32::from_rgb(0, 255, 255);
const ROI_COLOR: Color32 = Color32::from_rgb(255, 255, 255);
const EYE_COLOR: Color32 = Color32::from_rgb(0, 255, 0);
const LANDMARK_COLOR: Color32 = Color32::from_rgb(255, 255, 0);
const XYZ_COLOR: Color32 = Color32::from_rgb(255, 180, 0);
const EYE_RADIUS: f32 = 6.0;
const LANDMARK_RADIUS: f32 = 3.0;

/// Draw face detections with optional landmarks and XYZ coordinates on the egui painter.
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

        // Confidence label above the box
        let label = format!("{:.0}%", face.confidence * 100.0);
        painter.text(
            rect.left_top() + egui::vec2(2.0, -14.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::proportional(12.0),
            BBOX_COLOR,
        );

        // XYZ coordinates below the confidence label (inside the box top)
        if let Some([x, y, z]) = face.xyz {
            let xyz_label = format!("({:.2},{:.2},{:.2})m", x, y, z);
            painter.text(
                rect.left_top() + egui::vec2(2.0, 2.0),
                egui::Align2::LEFT_TOP,
                xyz_label,
                egui::FontId::monospace(10.0),
                XYZ_COLOR,
            );
        }

        // Draw landmarks (if available — head-only detections may not have them)
        if let Some(ref landmarks) = face.landmarks {
            for (i, lm) in landmarks.iter().enumerate() {
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
}

/// Draw the ROI crop boundary as a dashed-style rectangle.
pub fn draw_roi(
    painter: &Painter,
    roi_rect: Option<[u32; 4]>,
    offset: Pos2,
    scale_x: f32,
    scale_y: f32,
) {
    let Some([x1, y1, x2, y2]) = roi_rect else { return };
    let rect = egui::Rect::from_min_max(
        Pos2::new(
            offset.x + x1 as f32 * scale_x,
            offset.y + y1 as f32 * scale_y,
        ),
        Pos2::new(
            offset.x + x2 as f32 * scale_x,
            offset.y + y2 as f32 * scale_y,
        ),
    );
    painter.rect_stroke(
        rect,
        0.0,
        Stroke::new(1.0, ROI_COLOR.gamma_multiply(0.5)),
        StrokeKind::Outside,
    );
}
