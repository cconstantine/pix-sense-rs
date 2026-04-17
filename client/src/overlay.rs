use egui::{Color32, Painter, Pos2, Sense, Stroke, StrokeKind};
use pix_sense_common::FaceDetection;

const BBOX_COLOR: Color32 = Color32::from_rgb(0, 255, 255);
const TRACKED_COLOR: Color32 = Color32::from_rgb(0, 255, 100);
const UNTRACKED_COLOR: Color32 = Color32::from_rgb(120, 120, 120);
const ROI_COLOR: Color32 = Color32::from_rgb(255, 255, 255);
const EYE_COLOR: Color32 = Color32::from_rgb(0, 255, 0);
const LANDMARK_COLOR: Color32 = Color32::from_rgb(255, 255, 0);
const XYZ_COLOR: Color32 = Color32::from_rgb(255, 180, 0);
const EYE_RADIUS: f32 = 6.0;
const LANDMARK_RADIUS: f32 = 3.0;

/// Draw face detections with optional landmarks and XYZ coordinates.
/// When `tracked_idx` is `Some`, the face at that index is highlighted as the actively
/// tracked person and others are dimmed. Each face box is clickable — if the user
/// clicked one this frame, its index is returned.
///
/// `id_salt` scopes the per-face interaction IDs so callers can draw faces in
/// multiple panels (e.g. RGB and IR) without ID collisions.
pub fn draw_faces(
    ui: &mut egui::Ui,
    faces: &[FaceDetection],
    offset: Pos2,
    scale_x: f32,
    scale_y: f32,
    tracked_idx: Option<usize>,
    id_salt: &str,
) -> Option<usize> {
    let has_tracked = tracked_idx.is_some();
    let mut clicked: Option<usize> = None;
    let painter = ui.painter().clone();

    for (i, face) in faces.iter().enumerate() {
        let is_tracked = tracked_idx == Some(i);

        // Choose color based on tracking state
        let box_color = if is_tracked {
            TRACKED_COLOR
        } else if has_tracked {
            UNTRACKED_COLOR
        } else {
            BBOX_COLOR
        };
        let box_width = if is_tracked { 2.5 } else { 1.5 };

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

        // Clickable hit region — painted above the image, below the overlay lines.
        let response = ui.interact(
            rect,
            egui::Id::new((id_salt, i)),
            Sense::click(),
        );
        if response.clicked() {
            clicked = Some(i);
        }

        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(box_width, box_color),
            StrokeKind::Outside,
        );

        // Tracking label for the actively tracked person
        if is_tracked {
            painter.text(
                rect.left_top() + egui::vec2(2.0, -28.0),
                egui::Align2::LEFT_TOP,
                "TRACKING",
                egui::FontId::proportional(12.0),
                TRACKED_COLOR,
            );
        }

        // Confidence label above the box
        let label = format!("{:.0}%", face.confidence * 100.0);
        painter.text(
            rect.left_top() + egui::vec2(2.0, -14.0),
            egui::Align2::LEFT_TOP,
            label,
            egui::FontId::proportional(12.0),
            box_color,
        );

        // XYZ coordinates below the confidence label (inside the box top)
        if let Some([x, y, z]) = face.xyz {
            let xyz_label = format!("({:.2},{:.2},{:.2})m", x, y, z);
            painter.text(
                rect.left_top() + egui::vec2(2.0, 2.0),
                egui::Align2::LEFT_TOP,
                xyz_label,
                egui::FontId::monospace(10.0),
                if is_tracked { TRACKED_COLOR } else { XYZ_COLOR },
            );
        }

        // Draw landmarks (if available — head-only detections may not have them)
        if let Some(ref landmarks) = face.landmarks {
            for (li, lm) in landmarks.iter().enumerate() {
                let center = Pos2::new(
                    offset.x + lm.x * scale_x,
                    offset.y + lm.y * scale_y,
                );

                // Eyes get larger, green circles; other landmarks are smaller, yellow
                let (radius, color) = if li <= 1 {
                    (EYE_RADIUS, EYE_COLOR)
                } else {
                    (LANDMARK_RADIUS, LANDMARK_COLOR)
                };

                let alpha = if has_tracked && !is_tracked { 0.4 } else { 1.0 };
                painter.circle_filled(center, radius, color.gamma_multiply(alpha));
                painter.circle_stroke(center, radius, Stroke::new(1.0, Color32::BLACK));
            }
        }
    }

    clicked
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
