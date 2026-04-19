use std::time::{Duration, Instant};

use crate::fusion::TrackId;

const ROTATION_INTERVAL: Duration = Duration::from_secs(30);

pub struct PersonSelector {
    /// Stable ID of the actively-tracked person. As long as a track with this
    /// ID exists in the input, the selector returns it — independent of
    /// position jitter or which cameras contributed.
    current_id: Option<TrackId>,
    /// Last known position of the active track. Used only as a fallback when
    /// `current_id` is no longer present (e.g. the track expired after >1 s of
    /// misses), to pick a sensible replacement near where the person was.
    last_position: Option<[f32; 3]>,
    /// When we last pinned to the current ID. Drives the 30-second auto-rotate.
    locked_since: Instant,
}

impl PersonSelector {
    pub fn new() -> Self {
        Self {
            current_id: None,
            last_position: None,
            locked_since: Instant::now(),
        }
    }

    /// Lock tracking onto whichever track is closest to `target` (world frame).
    /// The next call to `select` resolves `target` to a concrete `TrackId` via
    /// the fallback path; the 30-second rotation timer also resets, so manual
    /// selections persist for one rotation window before auto-rotation resumes.
    pub fn lock_to(&mut self, target: [f32; 3]) {
        self.current_id = None;
        self.last_position = Some(target);
        self.locked_since = Instant::now();
    }

    /// Given all current tracks (id + world position), return the index and
    /// position of the single track to drive the renderer.
    pub fn select(&mut self, tracks: &[(TrackId, [f32; 3])]) -> Option<(usize, [f32; 3])> {
        if tracks.is_empty() {
            self.current_id = None;
            self.last_position = None;
            return None;
        }

        let now = Instant::now();

        // Sticky path: the active ID is still in the list — keep it, modulo
        // the 30 s auto-rotate.
        if let Some(id) = self.current_id {
            if let Some((idx, (_, pos))) =
                tracks.iter().enumerate().find(|(_, (tid, _))| *tid == id)
            {
                let pos = *pos;
                if now.duration_since(self.locked_since) >= ROTATION_INTERVAL && tracks.len() > 1 {
                    let (new_idx, &(new_id, new_pos)) = tracks
                        .iter()
                        .enumerate()
                        .filter(|(_, (tid, _))| *tid != id)
                        .max_by(|a, b| {
                            distance(pos, a.1 .1)
                                .partial_cmp(&distance(pos, b.1 .1))
                                .unwrap()
                        })
                        .unwrap();
                    self.current_id = Some(new_id);
                    self.last_position = Some(new_pos);
                    self.locked_since = now;
                    return Some((new_idx, new_pos));
                }
                self.last_position = Some(pos);
                return Some((idx, pos));
            }
        }

        // Fallback: no active ID, or its track is gone. Pick the track closest
        // to the last known position; if we have none, pick the first.
        let (idx, &(id, pos)) = match self.last_position {
            Some(prev) => tracks
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    distance(prev, a.1 .1)
                        .partial_cmp(&distance(prev, b.1 .1))
                        .unwrap()
                })
                .unwrap(),
            None => (0, &tracks[0]),
        };
        self.current_id = Some(id);
        self.last_position = Some(pos);
        self.locked_since = now;
        Some((idx, pos))
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}
