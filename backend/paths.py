"""Path helpers — remap stored camera paths after moving project between machines."""
from __future__ import annotations

import os
import sqlite3

from config import VIDEOS_DIR


def _basename(path: str) -> str:
    return os.path.basename(path.replace("\\", "/"))


def resolve_video_path(stored_path: str, camera_id: str = "") -> str | None:
    """Return a valid local video path, or None if not found."""
    if not stored_path:
        return None

    stored_path = os.path.normpath(stored_path)
    if os.path.isfile(stored_path):
        return stored_path

    # Same filename under current videos directory (handles Windows → Linux moves)
    name = _basename(stored_path)
    candidate = os.path.join(VIDEOS_DIR, name)
    if os.path.isfile(candidate):
        return candidate

    # Match camN prefix in filename (e.g. cam1_lab_sanitize.mp4)
    if camera_id and os.path.isdir(VIDEOS_DIR):
        prefix = f"{camera_id}_"
        for fname in sorted(os.listdir(VIDEOS_DIR)):
            if fname.lower().startswith(prefix.lower()):
                p = os.path.join(VIDEOS_DIR, fname)
                if os.path.isfile(p):
                    return p

    return None


def migrate_camera_video_paths(conn: sqlite3.Connection | None = None) -> int:
    """Fix broken video_path entries in SQLite after folder move. Returns count updated."""
    close = False
    if conn is None:
        from backend.db import get_db

        conn = get_db()
        close = True

    updated = 0
    rows = conn.execute("SELECT camera_id, video_path FROM cameras").fetchall()
    for row in rows:
        stored = row["video_path"]
        resolved = resolve_video_path(stored, row["camera_id"])
        if resolved and os.path.normpath(resolved) != os.path.normpath(stored):
            conn.execute(
                "UPDATE cameras SET video_path=? WHERE camera_id=?",
                (resolved, row["camera_id"]),
            )
            updated += 1

    if updated:
        conn.commit()
    if close:
        conn.close()
    return updated
