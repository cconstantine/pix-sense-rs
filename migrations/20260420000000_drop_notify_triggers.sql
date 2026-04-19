-- Remove the LISTEN/NOTIFY plumbing inherited from the multi-process pixpq era.
-- This server runs in a single process and signals itself directly through
-- in-process channels, so the DB-side triggers and trigger functions are dead weight.

DROP TRIGGER IF EXISTS notify_tracking_locations ON tracking_locations;
DROP TRIGGER IF EXISTS notify_sculpture_settings ON sculpture_settings;
DROP TRIGGER IF EXISTS notify_patterns ON patterns;

DROP FUNCTION IF EXISTS _notify_tracking_location_update();
DROP FUNCTION IF EXISTS _notify_sculpture_settings_update();
DROP FUNCTION IF EXISTS _notify_patterns_update();
