use chrono::Utc;
use pix_sense_common::{
    CameraExtrinsics, DetectionAlgo, DetectionConfig, Pattern, PatternUpdate, SculptureSettings,
    StreamSelection,
};
use sqlx::{PgPool, Row as _};

/// Initialise a connection pool from the `DATABASE_URL` environment variable,
/// then run pending migrations from the `migrations/` directory.
/// Returns `None` (with a warning) if the variable is unset, allowing the app
/// to run without a database configured.
pub async fn connect() -> Option<PgPool> {
    let url = match std::env::var("DATABASE_URL") {
        Ok(u) => u,
        Err(_) => {
            tracing::warn!("DATABASE_URL not set — location tracking disabled");
            return None;
        }
    };
    let pool = match sqlx::postgres::PgPoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await
    {
        Ok(pool) => pool,
        Err(e) => {
            tracing::error!("Failed to connect to postgres: {e:#}");
            return None;
        }
    };

    if let Err(e) = sqlx::migrate!("../migrations").run(&pool).await {
        tracing::error!("Database migration failed: {e:#}");
        return None;
    }

    tracing::info!("Connected to postgres — location tracking enabled");
    Some(pool)
}

/// Write a batch of XYZ detections to the `tracking_locations` table, following
/// the pixsense upsert-by-proximity + expiry logic.
///
/// For each `[x, y, z]`:
///   1. Try to UPDATE an existing row within 0.1 m that is less than 1 second old.
///   2. If no row was updated, INSERT a new one with a fresh UUID name.
/// Then DELETE all rows older than 1 second.
pub async fn write_detections(pool: &PgPool, detections: &[[f32; 3]]) {
    let now = Utc::now();
    let proximity: f64 = 0.1;

    for &[x, y, z] in detections {
        let (x, y, z) = (x as f64, y as f64, z as f64);

        // Try to update the nearest existing row within the proximity window.
        // FOR UPDATE SKIP LOCKED avoids serialization conflicts under concurrent writes.
        let result = sqlx::query(
            "UPDATE tracking_locations
             SET x = $1, y = $2, z = $3, updated_at = $4
             WHERE name = (
                 SELECT name FROM tracking_locations
                 WHERE updated_at > NOW() - INTERVAL '1 second'
                   AND sqrt((x - $1)^2 + (y - $2)^2 + (z - $3)^2) < $5
                 ORDER BY sqrt((x - $1)^2 + (y - $2)^2 + (z - $3)^2)
                 LIMIT 1
                 FOR UPDATE SKIP LOCKED
             )",
        )
        .bind(x)
        .bind(y)
        .bind(z)
        .bind(now)
        .bind(proximity)
        .execute(pool)
        .await;

        let rows_affected = result.map(|r| r.rows_affected()).unwrap_or(0);

        if rows_affected == 0 {
            let name = "pixo-16".to_string();
            if let Err(e) = sqlx::query(
                "INSERT INTO tracking_locations (name, x, y, z, updated_at)
                 VALUES ($1, $2, $3, $4, $5)
                 ON CONFLICT (name) DO UPDATE SET x = $2, y = $3, z = $4, updated_at = $5",
            )
            .bind(&name)
            .bind(x)
            .bind(y)
            .bind(z)
            .bind(now)
            .execute(pool)
            .await
            {
                tracing::warn!("DB insert error: {e:#}");
            }
        }
    }

    // Remove stale rows (no detection within the last second).
    if let Err(e) = sqlx::query(
        "DELETE FROM tracking_locations WHERE updated_at < NOW() - INTERVAL '1 second'",
    )
    .execute(pool)
    .await
    {
        tracing::warn!("DB expire error: {e:#}");
    }
}

/// Load the stored camera extrinsics for the given camera, or None if not calibrated.
pub async fn load_extrinsics(pool: &PgPool, camera_id: &str) -> Option<CameraExtrinsics> {
    let row = sqlx::query(
        "SELECT r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz \
         FROM camera_extrinsics WHERE camera_id = $1",
    )
    .bind(camera_id)
    .fetch_optional(pool)
    .await
    .ok()??;

    Some(CameraExtrinsics {
        r: [
            [row.get::<f32, _>("r00"), row.get::<f32, _>("r01"), row.get::<f32, _>("r02")],
            [row.get::<f32, _>("r10"), row.get::<f32, _>("r11"), row.get::<f32, _>("r12")],
            [row.get::<f32, _>("r20"), row.get::<f32, _>("r21"), row.get::<f32, _>("r22")],
        ],
        t: [row.get::<f32, _>("tx"), row.get::<f32, _>("ty"), row.get::<f32, _>("tz")],
    })
}

/// Upsert camera extrinsics for the given camera.
pub async fn save_extrinsics(
    pool: &PgPool,
    camera_id: &str,
    ext: &CameraExtrinsics,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO camera_extrinsics
             (camera_id, r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz,calibrated_at)
         VALUES ($1, $2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,now())
         ON CONFLICT (camera_id) DO UPDATE SET
             r00=$2,r01=$3,r02=$4,r10=$5,r11=$6,r12=$7,
             r20=$8,r21=$9,r22=$10,tx=$11,ty=$12,tz=$13,
             calibrated_at=now()",
    )
    .bind(camera_id)
    .bind(ext.r[0][0])
    .bind(ext.r[0][1])
    .bind(ext.r[0][2])
    .bind(ext.r[1][0])
    .bind(ext.r[1][1])
    .bind(ext.r[1][2])
    .bind(ext.r[2][0])
    .bind(ext.r[2][1])
    .bind(ext.r[2][2])
    .bind(ext.t[0])
    .bind(ext.t[1])
    .bind(ext.t[2])
    .execute(pool)
    .await?;
    Ok(())
}

/// Load all LED positions grouped by FadeCandy device, in index order.
/// Returns a list of (fadecandy_address, led_positions) pairs — one entry per device.
pub async fn load_leds(pool: &PgPool) -> Vec<(String, Vec<[f32; 3]>)> {
    let rows = match sqlx::query(
        "SELECT f.address, l.idx, l.x, l.y, l.z \
         FROM leds l \
         JOIN fadecandies f ON f.id = l.fadecandy_id \
         ORDER BY f.id, l.idx",
    )
    .fetch_all(pool)
    .await
    {
        Ok(rows) => rows,
        Err(e) => {
            tracing::warn!("load_leds query failed: {e:#}");
            return Vec::new();
        }
    };

    // Group by address, preserving insertion order.
    let mut result: Vec<(String, Vec<[f32; 3]>)> = Vec::new();
    for row in rows {
        let addr: String = row.get("address");
        let pos: [f32; 3] = [row.get("x"), row.get("y"), row.get("z")];
        if let Some(last) = result.last_mut() {
            if last.0 == addr {
                last.1.push(pos);
                continue;
            }
        }
        result.push((addr, vec![pos]));
    }
    result
}

/// Load the active pattern GLSL code and display settings for the named sculpture.
/// Returns `(glsl_code, brightness, gamma, overscan)` or `None` if no enabled
/// active pattern is found.
pub async fn load_active_pattern(
    pool: &PgPool,
    sculpture_name: &str,
) -> Option<(String, f32, f32, bool)> {
    let row = sqlx::query(
        "SELECT p.glsl_code, p.overscan, s.brightness, s.gamma \
         FROM sculpture_settings s \
         JOIN patterns p ON p.name = s.active_pattern \
         WHERE s.name = $1 AND p.enabled = true",
    )
    .bind(sculpture_name)
    .fetch_optional(pool)
    .await
    .ok()??;

    Some((
        row.get("glsl_code"),
        row.get("brightness"),
        row.get("gamma"),
        row.get("overscan"),
    ))
}

/// Load the persisted detection config, or return `None` if the row is missing.
pub async fn load_detection_config(pool: &PgPool) -> Option<DetectionConfig> {
    let row = sqlx::query("SELECT algo, stream FROM detection_config WHERE id = 1")
        .fetch_optional(pool)
        .await
        .ok()??;

    let algo = match row.get::<&str, _>("algo") {
        "yolo_head" => DetectionAlgo::YoloHead,
        "scrfd_face" => DetectionAlgo::ScrfdFace,
        "yolo_head_scrfd_landmarks" => DetectionAlgo::YoloHeadScrfdLandmarks,
        other => {
            tracing::warn!("Unknown algo in DB: '{other}' — using default");
            DetectionAlgo::YoloHead
        }
    };
    let stream = match row.get::<&str, _>("stream") {
        "rgb" => StreamSelection::Rgb,
        "ir" => StreamSelection::Ir,
        other => {
            tracing::warn!("Unknown stream in DB: '{other}' — using default");
            StreamSelection::Rgb
        }
    };
    Some(DetectionConfig { algo, stream })
}

/// Upsert the detection config (single-row table).
pub async fn save_detection_config(pool: &PgPool, cfg: DetectionConfig) {
    let algo = match cfg.algo {
        DetectionAlgo::YoloHead => "yolo_head",
        DetectionAlgo::ScrfdFace => "scrfd_face",
        DetectionAlgo::YoloHeadScrfdLandmarks => "yolo_head_scrfd_landmarks",
    };
    let stream = match cfg.stream {
        StreamSelection::Rgb => "rgb",
        StreamSelection::Ir => "ir",
    };
    if let Err(e) = sqlx::query(
        "INSERT INTO detection_config (id, algo, stream) VALUES (1, $1, $2)
         ON CONFLICT (id) DO UPDATE SET algo = $1, stream = $2",
    )
    .bind(algo)
    .bind(stream)
    .execute(pool)
    .await
    {
        tracing::warn!("Failed to save detection_config: {e:#}");
    }
}

/// Delete the stored extrinsics for the given camera.
pub async fn clear_extrinsics(pool: &PgPool, camera_id: &str) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM camera_extrinsics WHERE camera_id = $1")
        .bind(camera_id)
        .execute(pool)
        .await?;
    Ok(())
}

/// List all patterns ordered by name.
pub async fn list_patterns(pool: &PgPool) -> Vec<Pattern> {
    let rows = sqlx::query(
        "SELECT name, glsl_code, enabled, overscan FROM patterns ORDER BY name",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();
    rows.into_iter()
        .map(|r| Pattern {
            name: r.get("name"),
            glsl_code: r.get("glsl_code"),
            enabled: r.get("enabled"),
            overscan: r.get("overscan"),
        })
        .collect()
}

/// Fetch a single pattern by name, or None if it does not exist.
pub async fn get_pattern(pool: &PgPool, name: &str) -> Option<Pattern> {
    let row = sqlx::query(
        "SELECT name, glsl_code, enabled, overscan FROM patterns WHERE name = $1",
    )
    .bind(name)
    .fetch_optional(pool)
    .await
    .ok()??;
    Some(Pattern {
        name: row.get("name"),
        glsl_code: row.get("glsl_code"),
        enabled: row.get("enabled"),
        overscan: row.get("overscan"),
    })
}

/// Insert a new pattern. Returns a unique-violation error if the name already exists.
pub async fn create_pattern(pool: &PgPool, pattern: &Pattern) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO patterns (name, glsl_code, enabled, overscan) VALUES ($1, $2, $3, $4)",
    )
    .bind(&pattern.name)
    .bind(&pattern.glsl_code)
    .bind(pattern.enabled)
    .bind(pattern.overscan)
    .execute(pool)
    .await?;
    Ok(())
}

/// Update an existing pattern's code and settings.
/// Returns `true` if a row was updated, `false` if the name was not found.
pub async fn update_pattern(
    pool: &PgPool,
    name: &str,
    update: &PatternUpdate,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query(
        "UPDATE patterns SET glsl_code = $2, enabled = $3, overscan = $4 WHERE name = $1",
    )
    .bind(name)
    .bind(&update.glsl_code)
    .bind(update.enabled)
    .bind(update.overscan)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Delete a pattern by name.
/// Returns `true` if a row was deleted, `false` if the name was not found.
pub async fn delete_pattern(pool: &PgPool, name: &str) -> Result<bool, sqlx::Error> {
    let result = sqlx::query("DELETE FROM patterns WHERE name = $1")
        .bind(name)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

/// Set the active pattern for a sculpture, creating the settings row if it does not exist.
pub async fn set_active_pattern(
    pool: &PgPool,
    sculpture_name: &str,
    pattern_name: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO sculpture_settings (name, active_pattern, brightness, gamma)
         VALUES ($1, $2, 1.0, 2.2)
         ON CONFLICT (name) DO UPDATE SET active_pattern = $2",
    )
    .bind(sculpture_name)
    .bind(pattern_name)
    .execute(pool)
    .await?;
    Ok(())
}

/// Get the active pattern name for a sculpture, or None if no settings row exists.
pub async fn get_active_pattern(pool: &PgPool, sculpture_name: &str) -> Option<String> {
    let row = sqlx::query("SELECT active_pattern FROM sculpture_settings WHERE name = $1")
        .bind(sculpture_name)
        .fetch_optional(pool)
        .await
        .ok()??;
    Some(row.get("active_pattern"))
}

/// Fetch brightness/gamma for a sculpture. Returns None if no settings row exists.
pub async fn get_sculpture_settings(
    pool: &PgPool,
    sculpture_name: &str,
) -> Option<SculptureSettings> {
    let row = sqlx::query("SELECT brightness, gamma FROM sculpture_settings WHERE name = $1")
        .bind(sculpture_name)
        .fetch_optional(pool)
        .await
        .ok()??;
    Some(SculptureSettings {
        brightness: row.get("brightness"),
        gamma: row.get("gamma"),
    })
}

/// Update brightness/gamma for an existing sculpture. Returns `true` if a row
/// was updated. Does not create the row — a settings row is created lazily
/// by `set_active_pattern` on first activation.
pub async fn update_sculpture_settings(
    pool: &PgPool,
    sculpture_name: &str,
    settings: &SculptureSettings,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query(
        "UPDATE sculpture_settings SET brightness = $2, gamma = $3 WHERE name = $1",
    )
    .bind(sculpture_name)
    .bind(settings.brightness)
    .bind(settings.gamma)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}
