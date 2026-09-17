"""Food Industry Vision AI — Flask application."""
# Immediate feedback when run as a script (imports below can take several seconds).
if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("Loading Vision AI...", flush=True)

import hashlib
import json
import logging
import os
import sqlite3
import threading
from functools import wraps

from flask import Flask, Response, jsonify, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend import beta_ai
from backend.db import ensure_demo_user, get_db, init_db, seed_default_cameras
from backend.paths import migrate_camera_video_paths, resolve_video_path
from backend.detection_routes import register_detection_routes
from backend.alerts import delete_alert, delete_all_alerts
from backend.automode import apply_automode, get_automode_config, restore_automode_from_db
from backend.dashboard import dashboard_payload
from backend.mailer import (
    friendly_smtp_error,
    get_smtp_settings,
    save_smtp_settings,
    send_alert_email,
    send_test_email,
)
from backend.restore import restore_beta_from_db, restore_detections_from_db
from backend.runtime import (
    anpr_procs,
    automode_procs,
    beta_procs,
    entryexit_procs,
    fire_smoke_procs,
    flapgate_procs,
    fr_procs,
    headcount_procs,
    intrusion_procs,
    ppe_procs,
    video_readers,
    workforce_procs,
)
from backend.stream import get_stream_publisher, start_stream_publisher, stop_stream_publisher
from backend.video_reader import VideoFileReader
from backend.owlv2_scheduler import owlv2_scheduler_stats
from config import BASE_DIR, DISPLAY_FPS, HOST, PORT, VIDEOS_DIR, VLM_DEVICE, log_paths, pick_listen_port, server_access_urls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "frontend", "dist"), static_url_path="")
app.secret_key = os.environ.get("VISION_SECRET_KEY", "food-industry-demo-secret")
# Allow large video uploads (default Werkzeug limit is too small for many MP4s)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("VISION_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
CORS(app, supports_credentials=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".webm", ".mkv"}


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return wrapped


def _camera_row_to_dict(row, enabled_types=None):
    cid = row["camera_id"]
    try:
        rotation = int(row["rotation"] or 0)
    except (KeyError, IndexError, TypeError, ValueError):
        rotation = 0
    enabled = enabled_types or set()
    return {
        "camera_id": cid,
        "name": row["name"] or cid,
        "rtsp_url": row["video_path"],
        "status": row["status"] or "active",
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "location_label": row["location_label"],
        "rotation": rotation,
        # Runtime processors OR persisted enable flags (so AI Features Active counts stay correct)
        "headcount_active": cid in headcount_procs or ("headcount", cid) in enabled,
        "entryexit_active": cid in entryexit_procs or ("entryexit", cid) in enabled,
        "flapgate_active": cid in flapgate_procs or ("flapgate", cid) in enabled,
        "workforce_active": cid in workforce_procs or ("workforce", cid) in enabled,
        "intrusion_active": cid in intrusion_procs or ("intrusion", cid) in enabled,
        "ppe_active": cid in ppe_procs or ("ppe", cid) in enabled,
        "fire_smoke_active": cid in fire_smoke_procs or ("firesmoke", cid) in enabled,
        "anpr_active": cid in anpr_procs or ("anpr", cid) in enabled,
        "beta_active": cid in beta_procs,
        "automode_active": cid in automode_procs,
        "fr_active": cid in fr_procs or ("fr_detection", cid) in enabled,
    }


def _enabled_detection_pairs():
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT camera_id, detection_type FROM detection_configs WHERE enabled=1"
        ).fetchall()
        conn.close()
        return {(r["detection_type"], r["camera_id"]) for r in rows}
    except Exception:
        return set()


def _camera_rotation(row_or_id):
    if isinstance(row_or_id, str):
        conn = get_db()
        row = conn.execute("SELECT rotation FROM cameras WHERE camera_id=?", (row_or_id,)).fetchone()
        conn.close()
        try:
            return int(row["rotation"] or 0) if row else 0
        except (KeyError, IndexError, TypeError, ValueError):
            return 0
    try:
        return int(row_or_id["rotation"] or 0)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0


def _start_camera(camera_id, video_path, rotation=0):
    reader = VideoFileReader(camera_id, video_path, rotation=rotation)
    reader.start()
    video_readers[camera_id] = reader
    start_stream_publisher(camera_id)


def _stop_camera(camera_id):
    for registry in (
        beta_procs,
        headcount_procs,
        entryexit_procs,
        flapgate_procs,
        workforce_procs,
        intrusion_procs,
        ppe_procs,
        fire_smoke_procs,
        anpr_procs,
        fr_procs,
        automode_procs,
    ):
        if camera_id in registry:
            registry[camera_id].stop()
            del registry[camera_id]
    stop_stream_publisher(camera_id)
    if camera_id in video_readers:
        video_readers[camera_id].stop()
        del video_readers[camera_id]


def _discover_seed_videos():
    entries = []
    names = {
        "cam1": "Lab Sanitization",
        "cam2": "Production Line",
        "cam3": "Packaging Area",
        "cam4": "Industry PPE",
        "cam5": "Pharma PPE",
        "cam6": "Lab PPE Zone",
    }
    if os.path.isdir(VIDEOS_DIR):
        for i, fname in enumerate(sorted(os.listdir(VIDEOS_DIR)), start=1):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALLOWED_EXT:
                continue
            cid = f"cam{i}"
            entries.append(
                {
                    "camera_id": cid,
                    "name": names.get(cid, fname),
                    "video_path": os.path.join(VIDEOS_DIR, fname),
                    "location_label": f"Zone {i}",
                }
            )
    return entries[:6]


def _boot_cameras():
    conn = get_db()
    rows = conn.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()
    conn.close()
    for row in rows:
        if row["camera_id"] in video_readers:
            continue
        video_path = resolve_video_path(row["video_path"], row["camera_id"])
        if video_path and os.path.isfile(video_path):
            _start_camera(row["camera_id"], video_path, rotation=_camera_rotation(row))
        else:
            logger.warning(
                "Camera %s: video not found (stored: %s)",
                row["camera_id"],
                row["video_path"],
            )


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Invalid credentials"}), 401
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    if pwd_hash != row["password_hash"]:
        return jsonify({"error": "Invalid credentials"}), 401
    session["user_id"] = row["id"]
    session["username"] = row["username"]
    session["level"] = row["level"]
    return jsonify({"success": True, "username": row["username"], "level": row["level"]})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/auth/me")
def api_auth_me():
    if not session.get("user_id"):
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "username": session["username"], "level": session.get("level", "admin")})


# ── Cameras ───────────────────────────────────────────────────────────────────

@app.route("/api/cameras")
@login_required
def api_list_cameras():
    conn = get_db()
    rows = conn.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()
    conn.close()
    enabled = _enabled_detection_pairs()
    return jsonify([_camera_row_to_dict(r, enabled) for r in rows])


def _next_camera_id(conn):
    existing = {r["camera_id"] for r in conn.execute("SELECT camera_id FROM cameras").fetchall()}
    existing |= set(video_readers.keys())
    i = 1
    while f"cam{i}" in existing:
        i += 1
    return f"cam{i}"


def _ensure_rotation_column(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(cameras)").fetchall()}
    if "rotation" not in cols:
        conn.execute("ALTER TABLE cameras ADD COLUMN rotation INTEGER DEFAULT 0")
        conn.commit()


@app.route("/api/cameras", methods=["POST"])
@login_required
def api_add_camera():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    camera_id = (data.get("camera_id") or "").strip()
    video_path = (data.get("video_path") or data.get("rtsp_url") or "").strip()
    try:
        rotation = int(data.get("rotation", 0) or 0) % 360
    except (TypeError, ValueError):
        rotation = 0
    if rotation not in (0, 90, 180, 270):
        rotation = 0
    if not video_path:
        return jsonify({"error": "video_path required"}), 400

    conn = get_db()
    try:
        _ensure_rotation_column(conn)
        if not camera_id:
            camera_id = _next_camera_id(conn)
        exists = conn.execute("SELECT 1 FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
        if exists:
            return jsonify({"error": f"Camera ID '{camera_id}' already exists — pick another ID"}), 409
        if not os.path.isfile(video_path):
            return jsonify({"error": f"Video file not found: {video_path}"}), 400
        conn.execute(
            "INSERT INTO cameras (camera_id, name, video_path, status, rotation) VALUES (?,?,?,?,?)",
            (camera_id, name or camera_id, video_path, "active", rotation),
        )
        conn.commit()
    except Exception as e:
        logger.exception("Add camera failed")
        try:
            conn.rollback()
        except Exception:
            pass
        return jsonify({"error": f"Failed to add camera: {e}"}), 500
    finally:
        conn.close()

    try:
        _start_camera(camera_id, video_path, rotation=rotation)
    except Exception as e:
        logger.exception("Camera %s saved but failed to start playback", camera_id)
        return jsonify({
            "success": True,
            "camera_id": camera_id,
            "warning": f"Saved but playback failed to start: {e}",
        })
    return jsonify({"success": True, "camera_id": camera_id})


@app.route("/api/cameras/<camera_id>", methods=["DELETE"])
@login_required
def api_remove_camera(camera_id):
    _stop_camera(camera_id)
    conn = get_db()
    conn.execute("DELETE FROM cameras WHERE camera_id=?", (camera_id,))
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.route("/api/cameras/<camera_id>", methods=["PUT"])
@login_required
def api_update_camera(camera_id):
    """Update camera fields (rotation, name). Restarts reader if video/rotation changes."""
    data = request.json or {}
    conn = get_db()
    try:
        _ensure_rotation_column(conn)
        row = conn.execute("SELECT * FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
        if not row:
            return jsonify({"error": "Camera not found"}), 404

        name = data.get("name", row["name"])
        try:
            cur_rot = int(row["rotation"] or 0) if "rotation" in row.keys() else 0
        except (TypeError, ValueError, KeyError, IndexError):
            cur_rot = 0
        rotation = int(data.get("rotation", cur_rot) or 0) % 360
        if rotation not in (0, 90, 180, 270):
            rotation = 0
        conn.execute(
            "UPDATE cameras SET name=?, rotation=? WHERE camera_id=?",
            (name, rotation, camera_id),
        )
        conn.commit()
        video_path = resolve_video_path(row["video_path"], camera_id)
    except Exception as e:
        logger.exception("Update camera failed")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

    if camera_id in video_readers and video_path and os.path.isfile(video_path):
        old_reader = video_readers.get(camera_id)
        if old_reader:
            old_reader.stop()
        reader = VideoFileReader(camera_id, video_path, rotation=rotation)
        reader.start()
        video_readers[camera_id] = reader
        for registry in (
            headcount_procs,
            entryexit_procs,
            flapgate_procs,
            ppe_procs,
            fire_smoke_procs,
            anpr_procs,
            beta_procs,
            fr_procs,
        ):
            proc = registry.get(camera_id)
            if proc is not None and hasattr(proc, "camera_reader"):
                proc.camera_reader = reader

    return jsonify({"success": True, "rotation": rotation})


@app.route("/api/cameras/upload", methods=["POST"])
@login_required
def api_upload_video():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return jsonify({
                "error": f"Unsupported video format '{ext}'. Use: {', '.join(sorted(ALLOWED_EXT))}",
            }), 400
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        safe = secure_filename(f.filename) or f"upload{ext}"
        if not os.path.splitext(safe)[1]:
            safe = f"{safe}{ext}"
        # Avoid overwrite collisions
        base, e = os.path.splitext(safe)
        dest = os.path.join(VIDEOS_DIR, safe)
        n = 1
        while os.path.exists(dest):
            safe = f"{base}_{n}{e}"
            dest = os.path.join(VIDEOS_DIR, safe)
            n += 1
        f.save(dest)
        if not os.path.isfile(dest) or os.path.getsize(dest) <= 0:
            return jsonify({"error": "Upload failed — empty file written"}), 500
        logger.info("Uploaded video %s (%s bytes)", dest, os.path.getsize(dest))
        return jsonify({"success": True, "video_path": dest, "filename": safe})
    except Exception as e:
        logger.exception("Video upload failed")
        return jsonify({"error": f"Upload failed: {e}"}), 500


@app.route("/api/feed_snapshot/<camera_id>")
@login_required
def api_feed_snapshot(camera_id):
    pub = get_stream_publisher(camera_id)
    if pub is None:
        return jsonify({"error": "Camera not found"}), 404
    jpeg = pub.get_jpeg()
    if jpeg is None:
        return Response(b"", status=204, mimetype="image/jpeg")
    return Response(jpeg, mimetype="image/jpeg")


@app.route("/video_feed/live/<camera_id>")
@login_required
def video_feed_live(camera_id):
    interval = 1.0 / max(1.0, DISPLAY_FPS)

    def generate():
        import time

        while True:
            pub = get_stream_publisher(camera_id)
            if pub:
                part = pub.get_jpeg()
                if part:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + part + b"\r\n"
            time.sleep(interval)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Beta detection ────────────────────────────────────────────────────────────

@app.route("/api/detection/beta/config")
@login_required
def api_beta_config():
    conn = get_db()
    rows = conn.execute("SELECT * FROM beta_detection_prompts ORDER BY id").fetchall()
    beta_model = "owlv2"
    confidence = 0.35
    schedule_enabled = False
    schedule_start = ""
    schedule_end = ""
    try:
        s = conn.execute("SELECT * FROM beta_settings WHERE id=1").fetchone()
        if s:
            beta_model = s["model"] or "owlv2"
            if s["confidence"] is not None:
                confidence = float(s["confidence"])
            if "schedule_enabled" in s.keys():
                schedule_enabled = bool(s["schedule_enabled"])
            if "schedule_start" in s.keys():
                schedule_start = s["schedule_start"] or ""
            if "schedule_end" in s.keys():
                schedule_end = s["schedule_end"] or ""
    except sqlite3.OperationalError:
        pass
    conn.close()
    prompts = []
    for r in rows:
        try:
            cids = json.loads(r["camera_ids"]) if r["camera_ids"] else []
        except Exception:
            cids = []
        prompts.append({"id": r["id"], "prompt_text": r["prompt_text"] or "", "camera_ids": cids, "confidence": confidence})
    if not prompts:
        prompts = [{"prompt_text": "", "camera_ids": []}]
    return jsonify(
        {
            "prompts": prompts,
            "confidence": confidence,
            "beta_model": beta_model,
            "schedule_enabled": schedule_enabled,
            "schedule_start": schedule_start,
            "schedule_end": schedule_end,
            **beta_ai.beta_model_status(),
        }
    )


@app.route("/api/detection/beta/save", methods=["POST"])
@login_required
def api_beta_save():
    try:
        data = request.json or {}
        prompts_data = data.get("prompts", [])
        confidence = float(data.get("confidence", 0.2))
        backend = (data.get("model") or "owlv2").strip().lower()
        if backend not in ("owlv2", "qwen"):
            backend = "owlv2"
        if backend == "owlv2" and not beta_ai.owlv2_available():
            st = beta_ai.beta_model_status()
            return jsonify({"error": st.get("owlv2_message", "OWLv2 not available")}), 400
        if backend == "qwen" and not beta_ai.qwen_available():
            st = beta_ai.beta_model_status()
            return jsonify({"error": st.get("qwen_message", "Qwen not available.")}), 400

        conn = get_db()
        conn.execute("DELETE FROM beta_detection_prompts")
        for p in prompts_data:
            pt = (p.get("prompt_text") or "").strip()
            cids = p.get("camera_ids") or []
            if not pt and not cids:
                continue
            conn.execute(
                "INSERT INTO beta_detection_prompts (prompt_text, camera_ids, confidence, enabled) VALUES (?,?,?,1)",
                (pt, json.dumps(cids), confidence),
            )
        conn.execute(
            "INSERT OR REPLACE INTO beta_settings (id, model, confidence, schedule_enabled, schedule_start, schedule_end) VALUES (1, ?, ?, ?, ?, ?)",
            (
                backend,
                confidence,
                1 if bool(data.get("schedule_enabled")) else 0,
                str(data.get("schedule_start") or ""),
                str(data.get("schedule_end") or ""),
            ),
        )
        conn.commit()
        conn.close()

        camera_to_prompts = {}
        for p in prompts_data:
            pt = (p.get("prompt_text") or "").strip()
            for cid in p.get("camera_ids") or []:
                camera_to_prompts.setdefault(cid, []).append(pt)

        errors = []
        for cid in camera_to_prompts:
            if cid not in video_readers:
                errors.append({"camera_id": cid, "status": "camera offline"})

        from backend.schedule_util import is_schedule_active
        due = is_schedule_active({
            "schedule_enabled": bool(data.get("schedule_enabled")),
            "schedule_start": data.get("schedule_start") or "",
            "schedule_end": data.get("schedule_end") or "",
        })
        if due:
            _start_beta_processors(camera_to_prompts, confidence, backend)
            started = [cid for cid in camera_to_prompts if cid in video_readers and cid in beta_procs]
        else:
            for cid in list(beta_procs.keys()):
                try:
                    beta_procs[cid].stop()
                except Exception:
                    pass
                beta_procs.pop(cid, None)
            started = []

        msg = beta_ai.beta_model_status()
        if backend == "owlv2" and msg.get("owlv2_will_download"):
            note = "First run may take 1–2 minutes while OWLv2 downloads (~580MB)."
        elif backend == "owlv2":
            dev = str(msg.get("vlm_device") or "")
            if "cuda" in dev:
                note = "OWLv2 is running on GPU."
            else:
                note = "OWLv2 inference runs on CPU — allow 30–60s for first detections."
        else:
            note = "Qwen instruction monitoring started."

        return jsonify({
            "success": True,
            "started_cameras": started,
            "start_errors": errors,
            "backend": backend,
            "note": note,
        })
    except Exception as e:
        logger.exception("Beta save failed")
        return jsonify({"error": f"Beta configuration failed: {e}"}), 500


# ── Demo / stub APIs ──────────────────────────────────────────────────────────

@app.route("/api/stats")
@login_required
def api_stats():
    return jsonify(dashboard_payload())


@app.route("/api/dashboard")
@login_required
def api_dashboard():
    return jsonify(dashboard_payload())


@app.route("/api/alerts")
@login_required
def api_alerts():
    dtype = request.args.get("detection_type", "").strip()
    conn = get_db()
    if dtype:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE detection_type=? ORDER BY id DESC LIMIT 200",
            (dtype,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 200").fetchall()
    conn.close()
    alerts = []
    for r in rows:
        item = dict(r)
        snap = item.get("snapshot_path") or ""
        item["snapshot_url"] = f"/api/alerts/{item['id']}/snapshot" if snap else None
        alerts.append(item)
    return jsonify({"alerts": alerts})


@app.route("/api/alerts", methods=["DELETE"])
@login_required
def api_alerts_delete_all():
    delete_all_alerts()
    return jsonify({"ok": True})


@app.route("/api/alerts/<int:alert_id>", methods=["DELETE"])
@login_required
def api_alert_delete(alert_id):
    if not delete_alert(alert_id):
        return jsonify({"error": "Alert not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/alerts/<int:alert_id>/assign", methods=["POST"])
@login_required
def api_alert_assign(alert_id):
    data = request.get_json(silent=True) or {}
    try:
        engineer_id = int(data.get("engineer_id") or 0)
    except (TypeError, ValueError):
        engineer_id = 0
    if not engineer_id:
        return jsonify({"error": "Select an engineer"}), 400
    conn = get_db()
    alert = conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
    eng = conn.execute("SELECT * FROM engineers WHERE id=?", (engineer_id,)).fetchone()
    if not alert:
        conn.close()
        return jsonify({"error": "Alert not found"}), 404
    if not eng:
        conn.close()
        return jsonify({"error": "Engineer not found. Add them in System → Local."}), 404
    sent, err = send_alert_email(eng["email"], eng["name"], dict(alert))
    err = err or ""
    conn.execute(
        "INSERT INTO alert_assignments (alert_id, engineer_id, email_sent, email_error) VALUES (?,?,?,?)",
        (alert_id, engineer_id, 1 if sent else 0, err),
    )
    try:
        conn.execute("UPDATE alerts SET assigned_engineer_id=? WHERE id=?", (engineer_id, alert_id))
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "email_sent": sent, "email_error": err or ""})


@app.route("/api/alerts/<int:alert_id>/snapshot")
@login_required
def api_alert_snapshot(alert_id):
    from config import ALERTS_DIR

    conn = get_db()
    row = conn.execute("SELECT snapshot_path FROM alerts WHERE id=?", (alert_id,)).fetchone()
    conn.close()
    if not row or not row["snapshot_path"]:
        return jsonify({"error": "No snapshot"}), 404
    fname = os.path.basename(row["snapshot_path"])
    path = os.path.join(ALERTS_DIR, fname)
    if not os.path.isfile(path):
        return jsonify({"error": "Snapshot file missing"}), 404
    return send_from_directory(ALERTS_DIR, fname, mimetype="image/jpeg")


@app.route("/api/automode/config", methods=["GET"])
@login_required
def api_automode_get():
    cfg = get_automode_config()
    cfg["active_cameras"] = list(automode_procs.keys())
    cfg["owlv2_available"] = beta_ai.owlv2_available()
    return jsonify(cfg)


@app.route("/api/automode/config", methods=["POST"])
@login_required
def api_automode_save():
    try:
        data = request.get_json(silent=True) or {}
        enabled = bool(data.get("enabled"))
        prompt = (data.get("prompt") or "").strip()
        camera_ids = data.get("camera_ids") or []
        if not isinstance(camera_ids, list):
            camera_ids = []
        confidence = float(data.get("confidence", 0.2))
        schedule_enabled = bool(data.get("schedule_enabled"))
        schedule_start = str(data.get("schedule_start") or "")
        schedule_end = str(data.get("schedule_end") or "")
        result = apply_automode(
            enabled, prompt, camera_ids, confidence, schedule_enabled, schedule_start, schedule_end
        )
        result["prompt"] = prompt
        result["camera_ids"] = camera_ids
        result["confidence"] = confidence
        result["schedule_enabled"] = schedule_enabled
        result["schedule_start"] = schedule_start
        result["schedule_end"] = schedule_end
        return jsonify(result)
    except Exception as e:
        logger.exception("Autotrack apply failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/system/response-engineers", methods=["GET"])
@login_required
def api_engineers():
    conn = get_db()
    rows = conn.execute("SELECT * FROM engineers ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    return jsonify({"engineers": [dict(r) for r in rows]})


@app.route("/api/system/response-engineers", methods=["POST"])
@login_required
def api_engineers_add():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    department = (data.get("department") or "").strip()
    email = (data.get("email") or "").strip()
    if not name or not email:
        return jsonify({"error": "Name and email are required"}), 400
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO engineers (name, department, email) VALUES (?,?,?)",
        (name, department, email),
    )
    conn.commit()
    eid = cur.lastrowid
    conn.close()
    return jsonify({"ok": True, "id": eid, "name": name, "department": department, "email": email})


@app.route("/api/system/response-engineers/<int:engineer_id>", methods=["DELETE"])
@login_required
def api_engineers_delete(engineer_id):
    conn = get_db()
    conn.execute("DELETE FROM engineers WHERE id=?", (engineer_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/system/smtp", methods=["GET"])
@login_required
def api_smtp_get():
    cfg = get_smtp_settings()
    return jsonify({
        "from_email": cfg["from_email"],
        "smtp_host": cfg["smtp_host"],
        "smtp_port": cfg["smtp_port"],
        "has_password": bool(cfg["app_password"]),
    })


@app.route("/api/system/smtp", methods=["POST"])
@login_required
def api_smtp_save():
    data = request.get_json(silent=True) or {}
    save_smtp_settings(
        data.get("from_email") or "",
        data.get("app_password") or "",
        data.get("smtp_host") or "smtp.gmail.com",
        data.get("smtp_port") or 587,
    )
    return jsonify({"ok": True})


@app.route("/api/system/smtp/test", methods=["POST"])
@login_required
def api_smtp_test():
    data = request.get_json(silent=True) or {}
    if data.get("from_email") or data.get("app_password"):
        save_smtp_settings(
            data.get("from_email") or "",
            data.get("app_password") or "",
            data.get("smtp_host") or "smtp.gmail.com",
            data.get("smtp_port") or 587,
        )
    ok, err = send_test_email()
    if not ok:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True})


@app.route("/api/assigned-engineers")
@login_required
def api_assigned_engineers():
    conn = get_db()
    try:
        rows = conn.execute(
            """
            SELECT a.id, a.created_at, a.email_sent, a.email_error,
                   e.name AS engineer, e.department, e.email,
                   al.id AS alert_id, al.camera_id, al.detection_type, al.severity, al.message, al.created_at AS alert_time
            FROM alert_assignments a
            JOIN engineers e ON e.id = a.engineer_id
            JOIN alerts al ON al.id = a.alert_id
            ORDER BY a.id DESC
            LIMIT 200
            """
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    assignments = []
    for r in rows:
        assignments.append({
            "id": r["id"],
            "engineer": r["engineer"],
            "department": r["department"],
            "email": r["email"],
            "camera_id": r["camera_id"],
            "detection_type": r["detection_type"],
            "severity": r["severity"],
            "message": r["message"],
            "alert_time": r["alert_time"],
            "assigned_at": r["created_at"],
            "email_sent": bool(r["email_sent"]),
            "email_error": friendly_smtp_error(r["email_error"]) if r["email_error"] else "",
            "status": "Emailed" if r["email_sent"] else (friendly_smtp_error(r["email_error"]) if r["email_error"] else "Pending"),
        })
    return jsonify({"assignments": assignments})


@app.route("/api/plant-map")
@login_required
def api_plant_map():
    conn = get_db()
    rows = conn.execute("SELECT * FROM cameras").fetchall()
    conn.close()
    coords = [
        (12.9716, 77.5946),
        (12.9720, 77.5950),
        (12.9712, 77.5942),
        (12.9725, 77.5955),
        (12.9708, 77.5938),
        (12.9730, 77.5960),
    ]
    cameras = []
    for i, r in enumerate(rows):
        lat, lng = coords[i % len(coords)]
        cameras.append(
            {
                "camera_id": r["camera_id"],
                "name": r["name"],
                "latitude": r["latitude"] or lat,
                "longitude": r["longitude"] or lng,
                "location_label": r["location_label"] or f"Zone {i+1}",
                "status": r["status"],
                "health": "healthy",
                "alert_count": 0,
            }
        )
    return jsonify({"cameras": cameras, "plant_name": "Vision AI — Demo Plant"})


@app.route("/api/system-stats")
@login_required
def api_system_stats():
    import psutil

    return jsonify(
        {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "cameras_active": len(video_readers),
            "beta_active": len(beta_procs),
            "headcount_active": len(headcount_procs),
            "entryexit_active": len(entryexit_procs),
            "flapgate_active": len(flapgate_procs),
            "workforce_active": len(workforce_procs),
            "intrusion_active": len(intrusion_procs),
            "ppe_active": len(ppe_procs),
            "firesmoke_active": len(fire_smoke_procs),
            "anpr_active": len(anpr_procs),
            "fr_active": len(fr_procs),
            "automode_active": len(automode_procs),
            "owlv2_scheduler": owlv2_scheduler_stats(),
        }
    )


BUILD_ID = "2026-08-31-vision-ai-brand-v1"


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "app": "vision-ai",
        "build": BUILD_ID,
        "cameras": len(video_readers),
        "beta": len(beta_procs),
        "headcount": len(headcount_procs),
        "entryexit": len(entryexit_procs),
        "flapgate": len(flapgate_procs),
        "ppe": len(ppe_procs),
        "firesmoke": len(fire_smoke_procs),
        "anpr": len(anpr_procs),
        "fr": len(fr_procs),
        "workforce": len(workforce_procs),
        "intrusion": len(intrusion_procs),
        "automode": len(automode_procs),
        "frontend_dist": os.path.isfile(os.path.join(BASE_DIR, "frontend", "dist", "index.html")),
        "vlm_device_pref": VLM_DEVICE,
        "owlv2_scheduler": owlv2_scheduler_stats(),
    })


@app.route("/api/system/users")
@login_required
def api_users():
    return jsonify({"users": [{"id": 1, "username": "admin", "level": "admin", "active": True}]})


@app.route("/api/system/online-users")
@login_required
def api_online_users():
    return jsonify({"users": [{"username": session.get("username", "admin"), "since": "active"}]})


@app.route("/api/system/record-schedules")
@login_required
def api_record_schedules():
    return jsonify({"schedules": []})


@app.route("/api/system/holidays")
@login_required
def api_holidays():
    return jsonify({"holidays": []})


@app.route("/api/system/local-settings")
@login_required
def api_local_settings():
    return jsonify({"settings": {"snapshot_path": VIDEOS_DIR, "client_mode": "demo"}})


def _start_beta_processors(camera_to_prompts, confidence, backend):
    for cid in list(beta_procs.keys()):
        if cid not in camera_to_prompts:
            beta_procs[cid].stop()
            del beta_procs[cid]
    beta_ai.unload_beta_models_if_idle(len(beta_procs))
    for cid, prompts in camera_to_prompts.items():
        if cid not in video_readers:
            continue
        if cid in beta_procs:
            beta_procs[cid].stop()
        proc = beta_ai.make_beta_processor(cid, video_readers[cid], prompts, confidence, backend)
        proc.start()
        beta_procs[cid] = proc


def _ensure_camera_online(camera_id):
    if camera_id in video_readers:
        return True
    conn = get_db()
    row = conn.execute("SELECT * FROM cameras WHERE camera_id=?", (camera_id,)).fetchone()
    conn.close()
    if not row:
        return False
    video_path = resolve_video_path(row["video_path"], camera_id)
    if video_path and os.path.isfile(video_path):
        _start_camera(camera_id, video_path, rotation=_camera_rotation(row))
        return True
    return False


register_detection_routes(app, login_required, ensure_camera_fn=_ensure_camera_online)


# ── SPA fallback ──────────────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    dist = os.path.join(BASE_DIR, "frontend", "dist")
    if path and os.path.isfile(os.path.join(dist, path)):
        resp = send_from_directory(dist, path)
        # Hashed assets can be cached; keep them long-lived
        if path.startswith("assets/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return resp
    index = os.path.join(dist, "index.html")
    if os.path.isfile(index):
        resp = send_from_directory(dist, "index.html")
        # Always revalidate shell so new hashed JS/CSS is picked up after deploy
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        return resp
    return jsonify({"message": "Build frontend: cd frontend && npm install && npm run build"}), 503


def create_app():
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    init_db()
    ensure_demo_user()
    n = migrate_camera_video_paths()
    if n:
        logger.info("Migrated %d camera video path(s) to current server layout", n)
    seed_default_cameras(_discover_seed_videos())
    _boot_cameras()

    def _restore_all():
        restore_detections_from_db()
        restore_beta_from_db(_start_beta_processors)
        restore_automode_from_db()

    threading.Thread(target=_restore_all, daemon=True, name="RestoreDetections").start()
    from backend.schedule_supervisor import start_schedule_supervisor
    start_schedule_supervisor(start_beta_fn=_start_beta_processors)
    return app


if __name__ == "__main__":
    import traceback

    try:
        os.chdir(BASE_DIR)
        print("Starting server...", flush=True)
        create_app()

        listen_port = pick_listen_port(PORT)
        if os.environ.get("VISION_PORT", "").strip() == "8080":
            logger.warning(
                "VISION_PORT=8080 is set in your environment. "
                "This app uses port %s instead.",
                listen_port,
            )
        if listen_port != PORT:
            logger.warning("Port %s busy - using %s", PORT, listen_port)

        log_paths(logger)
        logger.info("Build: %s", BUILD_ID)
        logger.info(
            "Browser UI is served from frontend/dist — not frontend/src. "
            "After JSX/CSS changes run: cd frontend && npm run build"
        )
        dist_js = os.path.join(BASE_DIR, "frontend", "dist", "assets")
        if os.path.isdir(dist_js):
            assets = sorted(os.listdir(dist_js))
            logger.info("Frontend assets: %s", ", ".join(assets[:6]) if assets else "(empty - rebuild frontend!)")
        else:
            logger.warning("frontend/dist missing - AI Features UI will be stale/broken until you copy dist")
        urls = server_access_urls(listen_port)
        primary = urls[0] if urls else f"http://127.0.0.1:{listen_port}"
        logger.info("Vision AI - open %s  (login: admin / admin)", primary)
        print(f"Ready - open {primary}  (login: admin / admin)", flush=True)
        if len(urls) > 1:
            logger.info("Also available: %s", ", ".join(urls[1:]))
        app.run(host=HOST, port=listen_port, threaded=True, debug=False)
    except SystemExit as e:
        print(f"Startup aborted: {e}", flush=True)
        raise
    except Exception:
        print("FATAL: app failed to start:", flush=True)
        traceback.print_exc()
        sys.exit(1)
