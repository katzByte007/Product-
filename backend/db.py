"""SQLite database for cameras, beta config, and alerts."""
import json
import os
import sqlite3

from config import DATA_DIR, DB_PATH


def get_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id TEXT PRIMARY KEY,
            name TEXT,
            video_path TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            latitude REAL,
            longitude REAL,
            location_label TEXT,
            rotation INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS beta_detection_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_text TEXT,
            camera_ids TEXT,
            confidence REAL DEFAULT 0.2,
            enabled INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS beta_settings (
            id INTEGER PRIMARY KEY,
            model TEXT DEFAULT 'owlv2',
            confidence REAL DEFAULT 0.2
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT,
            detection_type TEXT,
            severity TEXT,
            message TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            status TEXT DEFAULT 'open'
        );
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            level TEXT DEFAULT 'admin'
        );
        CREATE TABLE IF NOT EXISTS detection_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            detection_type TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            config_json TEXT DEFAULT '{}',
            created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(camera_id, detection_type)
        );
        CREATE TABLE IF NOT EXISTS headcounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            current_count INTEGER DEFAULT 0,
            total_entries INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS entryexit_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            entries INTEGER DEFAULT 0,
            exits INTEGER DEFAULT 0,
            current_inside INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS flapgate_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            trespassing_count INTEGER DEFAULT 0,
            persons_in_frame INTEGER DEFAULT 0,
            gate1_status TEXT DEFAULT 'red',
            gate2_status TEXT DEFAULT 'red',
            gate3_status TEXT DEFAULT 'red'
        );
        CREATE TABLE IF NOT EXISTS ppe_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            compliant_count INTEGER DEFAULT 0,
            violation_count INTEGER DEFAULT 0,
            total_persons INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS fire_smoke_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            fire_count INTEGER DEFAULT 0,
            smoke_count INTEGER DEFAULT 0,
            total_detections INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS anpr_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            vehicle_type TEXT,
            plate_number TEXT,
            speed_kmh REAL,
            is_overspeeding INTEGER DEFAULT 0,
            track_id INTEGER,
            confidence REAL
        );
        CREATE TABLE IF NOT EXISTS smtp_settings (
            id INTEGER PRIMARY KEY,
            from_email TEXT DEFAULT '',
            app_password TEXT DEFAULT '',
            smtp_host TEXT DEFAULT 'smtp.gmail.com',
            smtp_port INTEGER DEFAULT 587
        );
        CREATE TABLE IF NOT EXISTS engineers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT DEFAULT '',
            email TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS alert_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id INTEGER NOT NULL,
            engineer_id INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            email_sent INTEGER DEFAULT 0,
            email_error TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS detection_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id TEXT,
            detection_type TEXT,
            stats_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS automode_settings (
            id INTEGER PRIMARY KEY,
            enabled INTEGER DEFAULT 0,
            prompt TEXT DEFAULT '',
            camera_ids TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0.2
        );
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO automode_settings (id, enabled, prompt, camera_ids, confidence) VALUES (1, 0, '', '[]', 0.2)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO smtp_settings (id, from_email, app_password, smtp_host, smtp_port) VALUES (1, '', '', 'smtp.gmail.com', 587)"
    )
    alert_cols = {r[1] for r in conn.execute("PRAGMA table_info(alerts)").fetchall()}
    if "snapshot_path" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN snapshot_path TEXT")
    if "assigned_engineer_id" not in alert_cols:
        conn.execute("ALTER TABLE alerts ADD COLUMN assigned_engineer_id INTEGER")
    conn.execute("INSERT OR IGNORE INTO beta_settings (id, model, confidence) VALUES (1, 'owlv2', 0.35)")
    # Migrate older DBs that lack rotation column
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cameras)").fetchall()}
    if "rotation" not in cols:
        conn.execute("ALTER TABLE cameras ADD COLUMN rotation INTEGER DEFAULT 0")
    auto_cols = {r[1] for r in conn.execute("PRAGMA table_info(automode_settings)").fetchall()}
    if "schedule_enabled" not in auto_cols:
        conn.execute("ALTER TABLE automode_settings ADD COLUMN schedule_enabled INTEGER DEFAULT 0")
    if "schedule_start" not in auto_cols:
        conn.execute("ALTER TABLE automode_settings ADD COLUMN schedule_start TEXT DEFAULT ''")
    if "schedule_end" not in auto_cols:
        conn.execute("ALTER TABLE automode_settings ADD COLUMN schedule_end TEXT DEFAULT ''")
    beta_cols = {r[1] for r in conn.execute("PRAGMA table_info(beta_settings)").fetchall()}
    if "schedule_enabled" not in beta_cols:
        conn.execute("ALTER TABLE beta_settings ADD COLUMN schedule_enabled INTEGER DEFAULT 0")
    if "schedule_start" not in beta_cols:
        conn.execute("ALTER TABLE beta_settings ADD COLUMN schedule_start TEXT DEFAULT ''")
    if "schedule_end" not in beta_cols:
        conn.execute("ALTER TABLE beta_settings ADD COLUMN schedule_end TEXT DEFAULT ''")
    conn.commit()
    conn.close()


def seed_default_cameras(video_entries):
    """Register cameras from video files if DB is empty."""
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) AS c FROM cameras").fetchone()["c"]
    if count > 0:
        conn.close()
        return
    for entry in video_entries:
        conn.execute(
            "INSERT INTO cameras (camera_id, name, video_path, status, location_label) VALUES (?,?,?,?,?)",
            (
                entry["camera_id"],
                entry["name"],
                entry["video_path"],
                "active",
                entry.get("location_label", ""),
            ),
        )
    conn.commit()
    conn.close()


def ensure_demo_user():
    import hashlib

    conn = get_db()
    row = conn.execute("SELECT id FROM users WHERE username=?", ("admin",)).fetchone()
    if row is None:
        pwd = hashlib.sha256(b"admin").hexdigest()
        conn.execute(
            "INSERT INTO users (username, password_hash, level) VALUES (?,?,?)",
            ("admin", pwd, "admin"),
        )
        conn.commit()
    conn.close()
