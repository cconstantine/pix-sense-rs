-- Idempotent migration: schema compatible with cconstantine/pixpq
-- Safe to run against an existing database — uses IF NOT EXISTS and ADD COLUMN IF NOT EXISTS.
-- Does not delete or truncate any data.

-- ---------------------------------------------------------------------------
-- tracking_locations
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tracking_locations (
    name       text      NOT NULL PRIMARY KEY,
    x          real      NOT NULL,
    y          real      NOT NULL,
    z          real      NOT NULL,
    tracked_at timestamp,
    updated_at timestamp NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- sculpture_settings
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sculpture_settings (
    name             text NOT NULL PRIMARY KEY,
    active_pattern   text NOT NULL,
    brightness       real NOT NULL,
    gamma            real NOT NULL
);

-- ---------------------------------------------------------------------------
-- patterns
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS patterns (
    name       text    NOT NULL PRIMARY KEY,
    glsl_code  text    NOT NULL,
    enabled    boolean NOT NULL,
    overscan   boolean NOT NULL DEFAULT true
);

ALTER TABLE patterns ADD COLUMN IF NOT EXISTS overscan boolean NOT NULL DEFAULT true;

-- ---------------------------------------------------------------------------
-- fadecandies
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fadecandies (
    id              SERIAL PRIMARY KEY,
    sculpture_name  text REFERENCES sculpture_settings(name),
    address         text UNIQUE NOT NULL
);

-- ---------------------------------------------------------------------------
-- leds
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS leds (
    fadecandy_id  integer REFERENCES fadecandies(id) NOT NULL,
    idx           integer NOT NULL,
    x             real    NOT NULL,
    y             real    NOT NULL,
    z             real    NOT NULL,
    PRIMARY KEY (fadecandy_id, idx)
);

-- ---------------------------------------------------------------------------
-- Triggers — NOTIFY on insert/update (compatible with pixpq LISTEN consumers)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION _notify_tracking_location_update() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('tracking_location_update', NEW.name);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS notify_tracking_locations ON tracking_locations;
CREATE TRIGGER notify_tracking_locations
    AFTER INSERT OR UPDATE ON tracking_locations
    FOR EACH ROW EXECUTE FUNCTION _notify_tracking_location_update();

CREATE OR REPLACE FUNCTION _notify_sculpture_settings_update() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('sculpture_settings_update', NEW.name);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS notify_sculpture_settings ON sculpture_settings;
CREATE TRIGGER notify_sculpture_settings
    AFTER INSERT OR UPDATE ON sculpture_settings
    FOR EACH ROW EXECUTE FUNCTION _notify_sculpture_settings_update();

CREATE OR REPLACE FUNCTION _notify_patterns_update() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('patterns_update', NEW.name);
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS notify_patterns ON patterns;
CREATE TRIGGER notify_patterns
    AFTER INSERT OR UPDATE ON patterns
    FOR EACH ROW EXECUTE FUNCTION _notify_patterns_update();
