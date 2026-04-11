use chrono::Utc;
use pix_sense_common::CameraExtrinsics;
use sqlx::{PgPool, Row as _};
use uuid::Uuid;

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
            let name = Uuid::new_v4().to_string();
            if let Err(e) = sqlx::query(
                "INSERT INTO tracking_locations (name, x, y, z, updated_at)
                 VALUES ($1, $2, $3, $4, $5)",
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

/// Delete the stored extrinsics for the given camera.
pub async fn clear_extrinsics(pool: &PgPool, camera_id: &str) -> Result<(), sqlx::Error> {
    sqlx::query("DELETE FROM camera_extrinsics WHERE camera_id = $1")
        .bind(camera_id)
        .execute(pool)
        .await?;
    Ok(())
}
