use sqlx::postgres::PgPoolOptions;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;

    if args.len() < 3 {
        eprintln!(
            "Usage: {} name LEDS_PER_SIDE [--reverse] [hostname]*",
            args[0]
        );
        std::process::exit(1);
    }

    let sculpture_name = args[i].clone();
    i += 1;
    let n: i32 = args[i].parse()?;
    i += 1;

    let mut reverse = false;
    if i < args.len() && args[i] == "--reverse" {
        reverse = true;
        i += 1;
    }
    let hostnames = &args[i..];

    let url = std::env::var("DATABASE_URL")
        .map_err(|_| anyhow::anyhow!("DATABASE_URL not set"))?;
    let pool = PgPoolOptions::new().max_connections(5).connect(&url).await?;
    sqlx::migrate!("../migrations").run(&pool).await?;

    // Ensure sculpture_settings row exists
    sqlx::query(
        "INSERT INTO sculpture_settings (name, active_pattern, brightness, gamma)
         VALUES ($1, '', 0.1, 1) ON CONFLICT (name) DO NOTHING",
    )
    .bind(&sculpture_name)
    .execute(&pool)
    .await?;

    // Ensure fadecandy rows exist and collect their ids
    let mut fadecandies: Vec<(i32, Vec<(f32, f32, f32)>)> = Vec::new();
    for host in hostnames {
        sqlx::query(
            "INSERT INTO fadecandies (sculpture_name, address) VALUES ($1, $2)
             ON CONFLICT (address) DO UPDATE SET sculpture_name = $1",
        )
        .bind(&sculpture_name)
        .bind(host)
        .execute(&pool)
        .await?;

        let (id,): (i32,) =
            sqlx::query_as("SELECT id FROM fadecandies WHERE address = $1")
                .bind(host)
                .fetch_one(&pool)
                .await?;

        fadecandies.push((id, Vec::new()));
    }

    // Generate LED positions — same math as pixo-creator.cpp
    // Coordinate mapping: db.x = x*spacing, db.y = z*spacing, db.z = y*spacing
    let spacing: f32 = 0.04318;
    let x_offset = -(n as f32) / 2.0 + 0.5;
    let y_offset = -(n as f32) / 2.0 + 0.5;
    let z_offset = -(n as f32) / 2.0 - 0.5;
    let per_fc = (n as usize) / fadecandies.len().max(1);

    for y in (1..=n).rev() {
        let mut dir: i32 = if reverse { 1 } else { -1 };
        let sel = ((y as usize) - 1) / per_fc.max(1);
        let sel = sel.min(fadecandies.len() - 1);

        for z in 0..n {
            // x_start replicates: std::max(-direction * (width - 1), 0)
            let x_start = (-dir * (n - 1)).max(0);
            let mut x = x_start;
            while x >= 0 && x < n {
                fadecandies[sel].1.push((
                    (x as f32 + x_offset) * spacing,
                    (z as f32 + y_offset) * spacing,
                    (y as f32 + z_offset) * spacing,
                ));
                x += dir;
            }
            dir *= -1;
        }
    }

    // Write to DB: delete old LEDs for each fadecandy, then insert new batch
    for (fc_id, leds) in &fadecandies {
        sqlx::query("DELETE FROM leds WHERE fadecandy_id = $1")
            .bind(fc_id)
            .execute(&pool)
            .await?;

        for (idx, &(x, y, z)) in leds.iter().enumerate() {
            sqlx::query(
                "INSERT INTO leds (fadecandy_id, idx, x, y, z) VALUES ($1, $2, $3, $4, $5)",
            )
            .bind(fc_id)
            .bind(idx as i32)
            .bind(x)
            .bind(y)
            .bind(z)
            .execute(&pool)
            .await?;
        }

        println!("fadecandy {} — {} LEDs inserted", fc_id, leds.len());
    }

    println!(
        "Done. Total LEDs: {}",
        fadecandies.iter().map(|(_, l)| l.len()).sum::<usize>()
    );

    Ok(())
}
