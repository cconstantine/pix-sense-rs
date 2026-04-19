-- Add auto-rotation interval to sculpture_settings.
-- 0 (the default) disables rotation; positive values rotate the active pattern
-- to a random enabled pattern every N minutes.

ALTER TABLE sculpture_settings
    ADD COLUMN IF NOT EXISTS rotation_minutes real NOT NULL DEFAULT 0;
