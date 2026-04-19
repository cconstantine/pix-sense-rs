//! Multi-camera detection fusion.
//!
//! Each camera thread emits a batch of world-frame detections per frame. The
//! fusion task greedy-matches them against existing tracks by world-frame
//! proximity, EMA-smooths track positions, expires stale tracks, and runs the
//! shared `PersonSelector` over the unified tracks list to pick one person to
//! drive the renderer.
//!
//! The selection round-trip is published back to each camera as a `TrackHint`
//! so the camera's encoder can stamp the correct `tracked_{rgb,ir}_idx` in the
//! outgoing frame metadata — and so calibration's `capture_tracked` handler has
//! the (cam-frame, world-frame) XYZ pair that camera contributed to the pick.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, watch};

use crate::person_selector::PersonSelector;

/// One detection from one camera, fully resolved (cam-frame + world-frame).
#[derive(Debug, Clone, Copy)]
pub struct FusionItem {
    /// Which stream the face index refers to.
    pub is_rgb: bool,
    /// Index into that stream's face list this frame.
    pub face_idx: usize,
    pub cam_xyz: [f32; 3],
    pub world_xyz: [f32; 3],
}

/// Per-frame message from a camera thread to the fusion task.
#[derive(Debug, Clone)]
pub struct CameraDetections {
    pub camera_id: String,
    pub items: Vec<FusionItem>,
}

/// Selection hint sent back to each camera per fusion round. `contributed` is
/// true iff this camera had a detection matched to the selected fused track.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrackHint {
    pub contributed: bool,
    pub tracked_rgb_idx: Option<usize>,
    pub tracked_ir_idx: Option<usize>,
    /// This camera's cam-frame XYZ contribution to the selected track (only
    /// meaningful when `contributed` is true).
    pub cam_xyz: [f32; 3],
    /// The fused (EMA-smoothed) world XYZ of the selected person.
    pub world_xyz: [f32; 3],
}

/// Greedy-NN threshold for matching a detection to an existing track.
const NN_THRESHOLD_M: f32 = 0.5;
/// A track is dropped if no camera has seen it for this long.
const TRACK_EXPIRE: Duration = Duration::from_secs(1);
/// EMA smoothing factor for track position updates (0 = hold, 1 = snap).
const EMA_ALPHA: f32 = 0.5;

#[derive(Debug)]
struct FusedTrack {
    world_pos: [f32; 3],
    last_seen: Instant,
    /// The newest contributor (by camera_id) from the most recent fusion round
    /// in which that camera saw this track. Cleared at the start of each round
    /// so only *this round's* contributors are visible when picking hints.
    contributors: HashMap<String, FusionItem>,
}

/// Spawn the fusion task on the current tokio runtime.
///
/// `hint_tx` holds one per-camera watch sender so each camera thread can read
/// the latest track hint via its matching receiver. `db_tx` is forwarded the
/// selected person's world XYZ (same semantics as the old single-camera path).
pub fn spawn(
    mut det_rx: mpsc::UnboundedReceiver<CameraDetections>,
    hint_tx: HashMap<String, watch::Sender<Option<TrackHint>>>,
    db_tx: mpsc::UnboundedSender<Vec<[f32; 3]>>,
    pending_selection: Arc<Mutex<Option<[f32; 3]>>>,
    running: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tracks: Vec<FusedTrack> = Vec::new();
        let mut selector = PersonSelector::new();

        while running.load(Ordering::Relaxed) {
            let det = match det_rx.recv().await {
                Some(d) => d,
                None => break,
            };

            // Apply any pending manual selection from the UI before running
            // the selector. Keeps the lock_to-then-select semantics of the
            // original single-camera loop.
            if let Some(target) = pending_selection.lock().unwrap().take() {
                selector.lock_to(target);
            }

            // Step 1: update/create tracks from this camera's detections.
            //
            // Matching is greedy NN against existing tracks. Detections that
            // are farther than NN_THRESHOLD_M from every track seed a new
            // track. Because we only see one camera per message, we clear just
            // this camera's slot in every track's contributors map — other
            // cameras' contributions persist until their own next message.
            let now = Instant::now();

            for t in tracks.iter_mut() {
                t.contributors.remove(&det.camera_id);
            }

            for item in det.items.iter() {
                // Find the nearest existing track within threshold.
                let mut best: Option<(usize, f32)> = None;
                for (i, t) in tracks.iter().enumerate() {
                    let d = distance(t.world_pos, item.world_xyz);
                    if d < NN_THRESHOLD_M {
                        match best {
                            Some((_, bd)) if bd <= d => {}
                            _ => best = Some((i, d)),
                        }
                    }
                }

                match best {
                    Some((idx, _)) => {
                        let t = &mut tracks[idx];
                        t.world_pos = ema(t.world_pos, item.world_xyz, EMA_ALPHA);
                        t.last_seen = now;
                        t.contributors.insert(det.camera_id.clone(), *item);
                    }
                    None => {
                        let mut contributors = HashMap::new();
                        contributors.insert(det.camera_id.clone(), *item);
                        tracks.push(FusedTrack {
                            world_pos: item.world_xyz,
                            last_seen: now,
                            contributors,
                        });
                    }
                }
            }

            // Step 2: expire tracks not seen by anyone in TRACK_EXPIRE.
            tracks.retain(|t| now.duration_since(t.last_seen) < TRACK_EXPIRE);

            // Step 3: run the global PersonSelector over the unified track
            // positions. The selector preserves its ≥30 s lock and 0.5 m
            // proximity semantics regardless of how many cameras contributed.
            let unified: Vec<[f32; 3]> = tracks.iter().map(|t| t.world_pos).collect();
            let picked = selector.select(&unified);

            // Step 4: fan out hints to all cameras.
            //
            // Every camera gets a hint every round — even those that didn't
            // contribute — so their watch receivers see the fresh `None` state
            // when tracking drops, preventing stale tracked_*_idx values from
            // sticking in the outgoing metadata.
            match picked {
                Some((track_idx, world_xyz)) => {
                    // Forward to the DB writer (and thus to the renderer via
                    // tracking_locations NOTIFY).
                    let _ = db_tx.send(vec![world_xyz]);

                    let track = &tracks[track_idx];
                    for (cam_id, tx) in hint_tx.iter() {
                        let hint = match track.contributors.get(cam_id) {
                            Some(item) => TrackHint {
                                contributed: true,
                                tracked_rgb_idx: if item.is_rgb { Some(item.face_idx) } else { None },
                                tracked_ir_idx: if !item.is_rgb { Some(item.face_idx) } else { None },
                                cam_xyz: item.cam_xyz,
                                world_xyz,
                            },
                            None => TrackHint {
                                contributed: false,
                                tracked_rgb_idx: None,
                                tracked_ir_idx: None,
                                cam_xyz: [0.0; 3],
                                world_xyz,
                            },
                        };
                        let _ = tx.send(Some(hint));
                    }
                }
                None => {
                    for tx in hint_tx.values() {
                        let _ = tx.send(None);
                    }
                }
            }
        }
    })
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn ema(prev: [f32; 3], new: [f32; 3], alpha: f32) -> [f32; 3] {
    [
        prev[0] + alpha * (new[0] - prev[0]),
        prev[1] + alpha * (new[1] - prev[1]),
        prev[2] + alpha * (new[2] - prev[2]),
    ]
}
