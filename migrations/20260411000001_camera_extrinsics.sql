-- Camera extrinsics: rigid body transform p_world = R * p_cam + t
-- Keyed by camera_id (RealSense serial number) so multiple cameras are supported.
CREATE TABLE IF NOT EXISTS camera_extrinsics (
    camera_id     text PRIMARY KEY,
    -- Row-major 3×3 rotation matrix
    r00 real NOT NULL, r01 real NOT NULL, r02 real NOT NULL,
    r10 real NOT NULL, r11 real NOT NULL, r12 real NOT NULL,
    r20 real NOT NULL, r21 real NOT NULL, r22 real NOT NULL,
    -- Translation vector (metres, world frame)
    tx  real NOT NULL,
    ty  real NOT NULL,
    tz  real NOT NULL,
    calibrated_at timestamptz NOT NULL DEFAULT now()
);
