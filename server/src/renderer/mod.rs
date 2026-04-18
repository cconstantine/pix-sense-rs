mod context;
mod opc;
mod pipeline;
mod shaders;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tokio::sync::watch;

use crate::db;
use pipeline::{RendererPipeline, FRAME_DURATION};

/// Entry point called from `main.rs` on a dedicated `std::thread`.
/// Never returns (loops until `running` is set to false).
pub fn run(
    pool: sqlx::PgPool,
    tracking_rx: watch::Receiver<Option<[f32; 3]>>,
    led_colors_tx: watch::Sender<Option<String>>,
    running: Arc<AtomicBool>,
    sculpture_name: String,
) {
    // Wrap in a restart loop so a crash reinitialises rather than killing the thread.
    loop {
        if !running.load(Ordering::Relaxed) {
            return;
        }
        if let Err(e) = run_inner(&pool, &tracking_rx, &led_colors_tx, &running, &sculpture_name) {
            tracing::error!("Renderer crashed: {e:#} — restarting in 5s");
            std::thread::sleep(std::time::Duration::from_secs(5));
        } else {
            return; // clean shutdown
        }
    }
}

fn run_inner(
    pool: &sqlx::PgPool,
    tracking_rx: &watch::Receiver<Option<[f32; 3]>>,
    led_colors_tx: &watch::Sender<Option<String>>,
    running: &Arc<AtomicBool>,
    sculpture_name: &str,
) -> anyhow::Result<()> {
    // Build a single-threaded Tokio runtime for async DB queries and OPC tasks.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // --- Load initial data from DB ---
    let (leds_by_device, initial_glsl, brightness, gamma, overscan) = rt.block_on(async {
        let leds_by_device = db::load_leds(pool).await;

        let (glsl, brightness, gamma, overscan) = db::load_active_pattern(pool, sculpture_name)
            .await
            .unwrap_or_else(|| {
                tracing::warn!(
                    "No active pattern found for sculpture '{sculpture_name}' — using black"
                );
                // Minimal valid pattern: output black.
                (
                    "void main() { fragColor = vec4(0.0, 0.0, 0.0, 1.0); }".to_string(),
                    1.0f32,
                    2.2f32,
                    true,
                )
            });

        tracing::info!(
            "Renderer: {} FadeCandy device(s), {} total LED(s), pattern loaded",
            leds_by_device.len(),
            leds_by_device.iter().map(|(_, l)| l.len()).sum::<usize>(),
        );

        anyhow::Ok((leds_by_device, glsl, brightness, gamma, overscan))
    })?;

    // Flatten all LEDs into one list for the VBO (order: device0 leds, device1 leds, …).
    let all_leds: Vec<[f32; 3]> = leds_by_device
        .iter()
        .flat_map(|(_, leds)| leds.iter().copied())
        .collect();

    // --- Shared state for hot-reload ---
    let reload_flag = Arc::new(AtomicBool::new(false));
    let new_glsl: Arc<Mutex<String>> = Arc::new(Mutex::new(initial_glsl.clone()));
    // (brightness, gamma, overscan)
    let new_render_settings: Arc<Mutex<(f32, f32, bool)>> =
        Arc::new(Mutex::new((brightness, gamma, overscan)));

    // --- Spawn settings watcher task (async, on the local rt) ---
    {
        let pool = pool.clone();
        let sn = sculpture_name.to_string();
        let flag = reload_flag.clone();
        let glsl_shared = new_glsl.clone();
        let settings_shared = new_render_settings.clone();
        let db_url = std::env::var("DATABASE_URL").unwrap_or_default();

        rt.spawn(async move {
            settings_watcher(pool, sn, db_url, flag, glsl_shared, settings_shared).await;
        });
    }

    // --- Spawn OPC sender tasks (async, on the local rt) ---
    let opc_senders: Vec<(tokio::sync::mpsc::Sender<Vec<u8>>, usize)> = {
        let _guard = rt.enter(); // set runtime context so tokio::spawn works
        let mut offset = 0;
        let mut senders = Vec::with_capacity(leds_by_device.len());
        for (addr, leds) in &leds_by_device {
            let tx = opc::spawn(addr.clone(), running.clone());
            senders.push((tx, offset));
            offset += leds.len();
        }
        senders
    };

    // --- Create headless EGL context ---
    let ctx = context::HeadlessContext::new()?;
    let gl = &ctx.gl;

    // --- Build GL pipeline ---
    let mut pipe = RendererPipeline::new(gl, &all_leds, &initial_glsl)?;
    let mut current_glsl = initial_glsl.clone();
    let mut current_brightness = brightness;
    let mut current_gamma = gamma;
    let mut current_overscan = overscan;

    tracing::info!("Renderer: GL pipeline initialised — entering frame loop");

    // --- Frame loop ---
    loop {
        let t_start = std::time::Instant::now();

        if !running.load(Ordering::Relaxed) {
            break;
        }

        // Check for shader/settings hot-reload. Only rebuild the shader if the
        // GLSL string actually changed — rebuilding resets the `time` uniform,
        // which visibly jerks the pattern on every brightness/gamma slider tick.
        if reload_flag.swap(false, Ordering::AcqRel) {
            let glsl = new_glsl.lock().unwrap().clone();
            let (b, g, o) = *new_render_settings.lock().unwrap();
            if glsl != current_glsl {
                pipe.reload_pattern_shader(gl, &glsl);
                current_glsl = glsl;
            }
            current_brightness = b;
            current_gamma = g;
            current_overscan = o;
        }

        // Get latest tracked person position.
        let location = *tracking_rx.borrow();

        // Pass 1: render GLSL pattern to 512×512 texture.
        pipe.render_pattern(gl, location.unwrap_or([0.0, 0.0, 0.0]));

        // Pass 2: project LED positions onto pattern (applies brightness,
        // gamma, and distance compensation in the shader).
        pipe.render_leds(
            gl,
            location,
            current_overscan,
            current_brightness,
            current_gamma,
        );

        // PBO readback (returns previous frame's data — 1-frame latency).
        if let Some(rgba) = pipe.pbo_readback(gl) {
            // Collect all LED RGB values in device order (matches /api/leds order).
            let mut all_rgb: Vec<[u8; 3]> = Vec::with_capacity(pipe.num_leds);

            // For each FadeCandy device, extract its LED slice, append to the
            // WebSocket broadcast buffer, and send via OPC.
            for (i, (tx, offset)) in opc_senders.iter().enumerate() {
                let next = opc_senders
                    .get(i + 1)
                    .map(|(_, o)| *o)
                    .unwrap_or(pipe.num_leds);
                let n_leds = next - offset;
                let rgb = RendererPipeline::extract_led_rgb(
                    &rgba[offset * 4..],
                    n_leds,
                );
                for chunk in rgb.chunks_exact(3) {
                    all_rgb.push([chunk[0], chunk[1], chunk[2]]);
                }
                let _ = tx.try_send(rgb); // drop frame on backpressure
            }

            // Broadcast LED colors to WebSocket clients via the watch channel.
            // Serialise as ServerMessage::LedColors.
            if !all_rgb.is_empty() {
                if let Ok(json) = serde_json::to_string(
                    &pix_sense_common::ServerMessage::LedColors(all_rgb),
                ) {
                    let _ = led_colors_tx.send(Some(json));
                }
            }
        }

        // Drive the Tokio reactor (OPC tasks, settings watcher) without blocking.
        rt.block_on(tokio::task::yield_now());

        // Sleep to target frame rate.
        let elapsed = t_start.elapsed();
        if elapsed < FRAME_DURATION {
            std::thread::sleep(FRAME_DURATION - elapsed);
        }
    }

    tracing::info!("Renderer: frame loop exited cleanly");
    Ok(())
}

// ---------------------------------------------------------------------------
// Settings watcher — listens for DB NOTIFY and reloads active pattern
// ---------------------------------------------------------------------------

async fn settings_watcher(
    pool: sqlx::PgPool,
    sculpture_name: String,
    db_url: String,
    reload_flag: Arc<AtomicBool>,
    new_glsl: Arc<Mutex<String>>,
    new_settings: Arc<Mutex<(f32, f32, bool)>>,
) {
    if db_url.is_empty() {
        return;
    }
    let mut listener = match sqlx::postgres::PgListener::connect(&db_url).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("settings_watcher: PgListener connect failed: {e:#}");
            return;
        }
    };
    if let Err(e) = listener
        .listen_all(["sculpture_settings_update", "patterns_update"])
        .await
    {
        tracing::warn!("settings_watcher: listen failed: {e:#}");
        return;
    }
    tracing::info!("settings_watcher: listening for pattern/settings changes");

    loop {
        match listener.recv().await {
            Ok(_) => {
                if let Some((glsl, b, g, o)) =
                    db::load_active_pattern(&pool, &sculpture_name).await
                {
                    *new_glsl.lock().unwrap() = glsl;
                    *new_settings.lock().unwrap() = (b, g, o);
                    reload_flag.store(true, Ordering::Release);
                    tracing::info!("settings_watcher: pattern/settings change detected — flagging reload");
                }
            }
            Err(e) => {
                tracing::warn!("settings_watcher: PgListener error: {e:#}");
            }
        }
    }
}
