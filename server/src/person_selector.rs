use std::time::{Duration, Instant};

const ROTATION_INTERVAL: Duration = Duration::from_secs(30);
const PROXIMITY_THRESHOLD: f32 = 0.5; // metres — match same person across frames

pub struct PersonSelector {
    /// Last known position of the currently tracked person.
    current_position: Option<[f32; 3]>,
    /// When we started tracking the current person.
    locked_since: Instant,
}

impl PersonSelector {
    pub fn new() -> Self {
        Self {
            current_position: None,
            locked_since: Instant::now(),
        }
    }

    /// Given all detected positions this frame, return the index and position of the
    /// single person to track.
    pub fn select(&mut self, detections: &[[f32; 3]]) -> Option<(usize, [f32; 3])> {
        if detections.is_empty() {
            self.current_position = None;
            return None;
        }

        let now = Instant::now();
        let elapsed = now.duration_since(self.locked_since);

        match self.current_position {
            Some(prev) => {
                // Find the detection closest to our currently tracked person.
                let (closest_idx, closest_dist) = detections
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (i, distance(prev, *d)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap(); // detections is non-empty

                if closest_dist >= PROXIMITY_THRESHOLD {
                    // Tracked person disappeared — switch immediately.
                    let picked = detections[0];
                    self.current_position = Some(picked);
                    self.locked_since = now;
                    Some((0, picked))
                } else if elapsed >= ROTATION_INTERVAL && detections.len() > 1 {
                    // Time to rotate — pick the person farthest from current.
                    let (new_idx, _) = detections
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != closest_idx)
                        .max_by(|a, b| {
                            distance(prev, *a.1)
                                .partial_cmp(&distance(prev, *b.1))
                                .unwrap()
                        })
                        .unwrap(); // len > 1 and we excluded one

                    let picked = detections[new_idx];
                    self.current_position = Some(picked);
                    self.locked_since = now;
                    Some((new_idx, picked))
                } else {
                    // Continue tracking the same person.
                    let picked = detections[closest_idx];
                    self.current_position = Some(picked);
                    Some((closest_idx, picked))
                }
            }
            None => {
                // No one was being tracked — pick the first detection.
                let picked = detections[0];
                self.current_position = Some(picked);
                self.locked_since = now;
                Some((0, picked))
            }
        }
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
