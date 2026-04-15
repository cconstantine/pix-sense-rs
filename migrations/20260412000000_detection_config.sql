-- Persists the detection algorithm and stream selection across restarts.
-- Single-row table enforced by CHECK (id = 1) + default PRIMARY KEY.
CREATE TABLE IF NOT EXISTS detection_config (
    id      integer PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    algo    text NOT NULL DEFAULT 'yolo_head',
    stream  text NOT NULL DEFAULT 'rgb'
);

-- Insert defaults on first run; do nothing if the row already exists.
INSERT INTO detection_config (id, algo, stream)
VALUES (1, 'yolo_head', 'rgb')
ON CONFLICT DO NOTHING;
