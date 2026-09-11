"""Alert persistence — snapshots + SQLite alerts table."""
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime

import cv2

from backend.db import get_db
from config import ALERTS_DIR, ALERT_COOLDOWN_SEC

logger = logging.getLogger(__name__)

_alert_lock = threading.Lock()
_alert_last_ts = {}


SKIP_ALERT_TYPES = {"entryexit", "headcount"}


def save_alert_event(camera_id, detection_type, frame, detections, severity="medium", meta=None, force=False, cooldown_sec=None):
    if frame is None:
        return
    if detection_type in SKIP_ALERT_TYPES:
        return
    now = time.time()
    key = (camera_id, detection_type)
    wait = ALERT_COOLDOWN_SEC if cooldown_sec is None else float(cooldown_sec)
    with _alert_lock:
        if not force and wait > 0 and now - _alert_last_ts.get(key, 0.0) < wait:
            return
        _alert_last_ts[key] = now

    labels = []
    if detections:
        for d in detections[:4]:
            if isinstance(d, (list, tuple)) and len(d) >= 2:
                labels.append(str(d[1]))
    message = ", ".join(labels) if labels else f"{detection_type} alert"
    if meta and meta.get("message"):
        message = str(meta["message"])[:500]

    os.makedirs(ALERTS_DIR, exist_ok=True)
    snapshot_path = ""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"{camera_id}_{detection_type}_{ts}.jpg".replace(":", "_")
        snapshot_path = os.path.join(ALERTS_DIR, fname)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 86])
        if ok:
            with open(snapshot_path, "wb") as f:
                f.write(buf.tobytes())
    except Exception as e:
        logger.debug("Alert snapshot save failed: %s", e)
        snapshot_path = ""

    rel_snapshot = os.path.basename(snapshot_path) if snapshot_path else ""

    try:
        conn = get_db()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(alerts)").fetchall()}
        if "snapshot_path" in cols:
            conn.execute(
                "INSERT INTO alerts (camera_id, detection_type, severity, message, status, snapshot_path) VALUES (?,?,?,?,?,?)",
                (camera_id, detection_type, severity, message[:500], "open", rel_snapshot or None),
            )
        else:
            conn.execute(
                "INSERT INTO alerts (camera_id, detection_type, severity, message, status) VALUES (?,?,?,?,?)",
                (camera_id, detection_type, severity, message[:500], "open"),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Alert DB insert failed %s/%s: %s", camera_id, detection_type, e)


def delete_alert(alert_id):
    conn = get_db()
    row = conn.execute("SELECT snapshot_path FROM alerts WHERE id=?", (alert_id,)).fetchone()
    if not row:
        conn.close()
        return False
    snap = row["snapshot_path"] or ""
    try:
        conn.execute("DELETE FROM alert_assignments WHERE alert_id=?", (alert_id,))
    except sqlite3.OperationalError:
        pass
    conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
    conn.commit()
    conn.close()
    if snap:
        fpath = os.path.join(ALERTS_DIR, os.path.basename(snap))
        try:
            if os.path.isfile(fpath):
                os.remove(fpath)
        except Exception:
            pass
    return True


def delete_all_alerts():
    conn = get_db()
    rows = conn.execute("SELECT snapshot_path FROM alerts").fetchall()
    try:
        conn.execute("DELETE FROM alert_assignments")
    except sqlite3.OperationalError:
        pass
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
    for row in rows:
        snap = row["snapshot_path"] or ""
        if not snap:
            continue
        fpath = os.path.join(ALERTS_DIR, os.path.basename(snap))
        try:
            if os.path.isfile(fpath):
                os.remove(fpath)
        except Exception:
            pass
    return True
