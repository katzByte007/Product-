"""Detection API routes — ported from Utthunga Dashboard.

Heavy YOLO/analytics imports are lazy so `python app.py` starts quickly.
"""
import json
import logging

from flask import jsonify, request

from backend.db import get_db
from backend.runtime import (
    anpr_procs,
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
from backend.schedule_util import attach_schedule, is_schedule_active
from backend.stub_detections import DummyDetectionProcessor
from config import (
    ENTRYEXIT_MODEL_PATH,
    FIRE_SMOKE_MODEL_PATH,
    FLAPGATE_MODEL_PATH,
    HEADCOUNT_MODEL_PATH,
    PPE_MODEL_PATH,
    WORKFORCE_MODEL_PATH,
)

logger = logging.getLogger(__name__)


def _dump_cfg(config, data):
    return json.dumps(attach_schedule(config, data or {}))


def _analytics():
    """Lazy-load YOLO processors (slow / may need GPU)."""
    from backend.analytics_runtime import (
        EntryExitProcessor,
        FireSmokeDetectionProcessor,
        FlapGateProcessor,
        HeadCountProcessor,
        PPEDetectionProcessor,
        get_inference_engine,
    )

    return {
        "EntryExitProcessor": EntryExitProcessor,
        "FireSmokeDetectionProcessor": FireSmokeDetectionProcessor,
        "FlapGateProcessor": FlapGateProcessor,
        "HeadCountProcessor": HeadCountProcessor,
        "PPEDetectionProcessor": PPEDetectionProcessor,
        "get_inference_engine": get_inference_engine,
    }


def register_detection_routes(app, login_required, ensure_camera_fn=None):
    """Register all detection + FR API routes on Flask app."""

    def _ensure_camera(camera_id):
        if camera_id in video_readers:
            return True
        if ensure_camera_fn:
            return ensure_camera_fn(camera_id)
        return False

    @app.route("/api/cameras/<camera_id>/snapshot")
    @login_required
    def api_camera_snapshot(camera_id):
        from backend.stream import get_stream_publisher
        from flask import Response
        import cv2

        pub = get_stream_publisher(camera_id)
        if pub is None:
            return jsonify({"error": "Camera not found"}), 404
        jpeg = pub.get_jpeg()
        if jpeg is None:
            reader = video_readers.get(camera_id)
            if reader is None:
                return jsonify({"error": "Camera offline"}), 404
            frame = reader.get_frame()
            if frame is None:
                return jsonify({"error": "No frame"}), 204
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return jsonify({"error": "Encode failed"}), 500
            return Response(buf.tobytes(), mimetype="image/jpeg")
        return Response(jpeg, mimetype="image/jpeg")

    @app.route("/api/detection/config/<camera_id>")
    @login_required
    def api_get_detection_config(camera_id):
        conn = get_db()
        rows = conn.execute("SELECT * FROM detection_configs WHERE camera_id=?", (camera_id,)).fetchall()
        conn.close()
        configs = {}
        for row in rows:
            configs[row["detection_type"]] = {
                "enabled": bool(row["enabled"]),
                "config": json.loads(row["config_json"]) if row["config_json"] else {},
            }
        return jsonify(configs)

    @app.route("/api/detection/configs/all")
    @login_required
    def api_all_detection_configs():
        conn = get_db()
        rows = conn.execute("SELECT camera_id, detection_type FROM detection_configs WHERE enabled=1").fetchall()
        conn.close()
        return jsonify([{"camera_id": r["camera_id"], "detection_type": r["detection_type"]} for r in rows])

    @app.route("/api/detection/headcount/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_enable_headcount(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        try:
            A = _analytics()
        except Exception as e:
            logger.exception("Failed to load analytics runtime")
            return jsonify({"error": f"Detection engine failed to load: {e}"}), 500
        data = request.json or {}
        conf = float(data.get("confidence", 0.1))
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "headcount", 1, _dump_cfg({"confidence": conf}, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in headcount_procs:
            headcount_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule({"confidence": conf}, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        try:
            engine = A["get_inference_engine"]("headcount", HEADCOUNT_MODEL_PATH)
            proc = A["HeadCountProcessor"](camera_id, video_readers[camera_id], engine, conf)
            headcount_procs[camera_id] = proc
            proc.start()
        except Exception as e:
            logger.exception("Enable headcount failed for %s", camera_id)
            headcount_procs.pop(camera_id, None)
            return jsonify({"error": f"Could not start headcount detection: {e}"}), 500
        return jsonify({"success": True})

    @app.route("/api/detection/headcount/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_disable_headcount(camera_id):
        if camera_id in headcount_procs:
            headcount_procs[camera_id].stop()
            del headcount_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "headcount"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/detection/entryexit/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_enable_entryexit(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        try:
            A = _analytics()
        except Exception as e:
            return jsonify({"error": f"Detection engine failed to load: {e}"}), 500
        data = request.json or {}
        entry_zone = data.get("entry_zone", [])
        exit_zone = data.get("exit_zone", [])
        canvas_w = int(data.get("canvas_width", 800))
        canvas_h = int(data.get("canvas_height", 450))
        conf = float(data.get("confidence", 0.5))
        if len(entry_zone) < 3 or len(exit_zone) < 3:
            return jsonify({"error": "Both zones need at least 3 points"}), 400
        entry_tuples = [(p["x"], p["y"]) if isinstance(p, dict) else tuple(p) for p in entry_zone]
        exit_tuples = [(p["x"], p["y"]) if isinstance(p, dict) else tuple(p) for p in exit_zone]
        config = {
            "entry_zone": [list(t) for t in entry_tuples],
            "exit_zone": [list(t) for t in exit_tuples],
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
            "confidence": conf,
        }
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "entryexit", 1, _dump_cfg(config, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in entryexit_procs:
            entryexit_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule(config, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        try:
            engine = A["get_inference_engine"]("entryexit", ENTRYEXIT_MODEL_PATH, classes=[0])
            proc = A["EntryExitProcessor"](
                camera_id, video_readers[camera_id], entry_tuples, exit_tuples, (canvas_w, canvas_h), engine, conf
            )
            entryexit_procs[camera_id] = proc
            proc.start()
        except Exception as e:
            logger.exception("Enable entryexit failed for %s", camera_id)
            entryexit_procs.pop(camera_id, None)
            return jsonify({"error": f"Could not start entry/exit detection: {e}"}), 500
        return jsonify({"success": True})

    @app.route("/api/detection/entryexit/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_disable_entryexit(camera_id):
        if camera_id in entryexit_procs:
            entryexit_procs[camera_id].stop()
            del entryexit_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "entryexit"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/detection/flapgate/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_enable_flapgate(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        try:
            A = _analytics()
        except Exception as e:
            return jsonify({"error": f"Detection engine failed to load: {e}"}), 500
        data = request.json or {}
        gate_zones_raw = data.get("gate_zones", {})
        canvas_w = int(data.get("canvas_width", 800))
        canvas_h = int(data.get("canvas_height", 450))
        conf = float(data.get("confidence", 0.65))
        gate_zones = {}
        for zid_str, pts in gate_zones_raw.items():
            zid = int(zid_str)
            if len(pts) < 3:
                return jsonify({"error": f"Gate {zid} zone needs at least 3 points"}), 400
            gate_zones[zid] = [(p["x"], p["y"]) if isinstance(p, dict) else tuple(p) for p in pts]
        if len(gate_zones) < 3:
            return jsonify({"error": "All 3 gate zones must be configured"}), 400
        config = {
            "gate_zones": {str(k): [list(p) for p in v] for k, v in gate_zones.items()},
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
            "confidence": conf,
        }
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "flapgate", 1, _dump_cfg(config, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in flapgate_procs:
            flapgate_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule(config, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        try:
            engine = A["get_inference_engine"]("flapgate", FLAPGATE_MODEL_PATH, classes=[0])
            proc = A["FlapGateProcessor"](camera_id, video_readers[camera_id], gate_zones, (canvas_w, canvas_h), engine, conf)
            flapgate_procs[camera_id] = proc
            proc.start()
        except Exception as e:
            logger.exception("Enable flapgate failed for %s", camera_id)
            flapgate_procs.pop(camera_id, None)
            return jsonify({"error": f"Could not start flap gate detection: {e}"}), 500
        return jsonify({"success": True})

    @app.route("/api/detection/flapgate/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_disable_flapgate(camera_id):
        if camera_id in flapgate_procs:
            flapgate_procs[camera_id].stop()
            del flapgate_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "flapgate"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/detection/ppe/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_enable_ppe(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        try:
            A = _analytics()
        except Exception as e:
            return jsonify({"error": f"Detection engine failed to load: {e}"}), 500
        data = request.json or {}
        conf = float(data.get("confidence", 0.4))
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "ppe", 1, _dump_cfg({"confidence": conf}, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in ppe_procs:
            ppe_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule({"confidence": conf}, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        try:
            engine = A["get_inference_engine"]("ppe", PPE_MODEL_PATH, classes=None)
            proc = A["PPEDetectionProcessor"](camera_id, video_readers[camera_id], engine, conf)
            ppe_procs[camera_id] = proc
            proc.start()
        except Exception as e:
            logger.exception("Enable ppe failed for %s", camera_id)
            ppe_procs.pop(camera_id, None)
            return jsonify({"error": f"Could not start PPE detection: {e}"}), 500
        return jsonify({"success": True})

    @app.route("/api/detection/ppe/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_disable_ppe(camera_id):
        if camera_id in ppe_procs:
            ppe_procs[camera_id].stop()
            del ppe_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "ppe"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/detection/firesmoke/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_enable_firesmoke(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        try:
            A = _analytics()
        except Exception as e:
            return jsonify({"error": f"Detection engine failed to load: {e}"}), 500
        data = request.json or {}
        conf = float(data.get("confidence", 0.35))
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "firesmoke", 1, _dump_cfg({"confidence": conf}, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in fire_smoke_procs:
            fire_smoke_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule({"confidence": conf}, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        try:
            engine = A["get_inference_engine"]("firesmoke", FIRE_SMOKE_MODEL_PATH, classes=None)
            proc = A["FireSmokeDetectionProcessor"](camera_id, video_readers[camera_id], engine, conf)
            fire_smoke_procs[camera_id] = proc
            proc.start()
        except Exception as e:
            logger.exception("Enable firesmoke failed for %s", camera_id)
            fire_smoke_procs.pop(camera_id, None)
            return jsonify({"error": f"Could not start fire/smoke detection: {e}"}), 500
        return jsonify({"success": True})

    @app.route("/api/detection/firesmoke/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_disable_firesmoke(camera_id):
        if camera_id in fire_smoke_procs:
            fire_smoke_procs[camera_id].stop()
            del fire_smoke_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "firesmoke"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/detection/anpr/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_enable_anpr(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        data = request.json or {}
        conf = float(data.get("confidence", 0.25))
        speed_threshold = float(data.get("speed_threshold_kmh", 50))
        meters_per_pixel = float(data.get("meters_per_pixel", 0.15))
        config = {
            "confidence": conf,
            "speed_threshold_kmh": speed_threshold,
            "meters_per_pixel": meters_per_pixel,
        }
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "anpr", 1, _dump_cfg(config, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in anpr_procs:
            anpr_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule(config, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        proc = DummyDetectionProcessor(camera_id, "anpr", config)
        anpr_procs[camera_id] = proc
        proc.start()
        return jsonify({"success": True, "demo": True})

    @app.route("/api/detection/anpr/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_disable_anpr(camera_id):
        if camera_id in anpr_procs:
            anpr_procs[camera_id].stop()
            del anpr_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "anpr"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/detection/config/<camera_id>/<detection_type>", methods=["PUT"])
    @login_required
    def api_put_detection_config(camera_id, detection_type):
        """Save zone/prompt config without starting detection (use Apply on AI Features to enable)."""
        data = request.json or {}
        conn = get_db()
        row = conn.execute(
            "SELECT enabled, config_json FROM detection_configs WHERE camera_id=? AND detection_type=?",
            (camera_id, detection_type),
        ).fetchone()
        enabled = int(row["enabled"]) if row else 0
        prev = {}
        if row and row["config_json"]:
            try:
                prev = json.loads(row["config_json"])
            except Exception:
                prev = {}
        prev.update(data)
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, detection_type, enabled, json.dumps(prev)),
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/detection/workforce/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_enable_workforce(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        try:
            A = _analytics()
        except Exception as e:
            return jsonify({"error": f"Detection engine failed to load: {e}"}), 500
        from backend.workforce_processor import WorkforceMonitoringProcessor

        data = request.json or {}
        machine_zone = data.get("machine_zone", [])
        canvas_w = int(data.get("canvas_width", 800))
        canvas_h = int(data.get("canvas_height", 450))
        conf = float(data.get("confidence", 0.25))
        away_alert_sec = float(data.get("away_alert_sec", 3))
        if len(machine_zone) < 3:
            return jsonify({"error": "Machine zone needs at least 3 points"}), 400
        zone_tuples = [(p["x"], p["y"]) if isinstance(p, dict) else tuple(p) for p in machine_zone]
        config = {
            "machine_zone": [list(t) for t in zone_tuples],
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
            "confidence": conf,
            "away_alert_sec": away_alert_sec,
        }
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "workforce", 1, _dump_cfg(config, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in workforce_procs:
            workforce_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule(config, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        try:
            engine = A["get_inference_engine"]("workforce", WORKFORCE_MODEL_PATH, classes=[0])
            proc = WorkforceMonitoringProcessor(
                camera_id, video_readers[camera_id], zone_tuples, (canvas_w, canvas_h), engine, conf,
                away_alert_sec=away_alert_sec,
            )
            workforce_procs[camera_id] = proc
            proc.start()
        except Exception as e:
            logger.exception("Enable workforce failed for %s", camera_id)
            workforce_procs.pop(camera_id, None)
            return jsonify({"error": f"Could not start workforce monitoring: {e}"}), 500
        return jsonify({"success": True})

    @app.route("/api/detection/workforce/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_disable_workforce(camera_id):
        if camera_id in workforce_procs:
            workforce_procs[camera_id].stop()
            del workforce_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "workforce"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/detection/intrusion/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_enable_intrusion(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        from backend.intrusion_processor import IntrusionDetectionProcessor
        import backend.beta_ai as beta_ai_mod

        data = request.json or {}
        zone = data.get("intrusion_zone") or data.get("zone") or []
        prompt = (data.get("prompt") or data.get("prompt_text") or "").strip()
        canvas_w = int(data.get("canvas_width", 800))
        canvas_h = int(data.get("canvas_height", 450))
        conf = float(data.get("confidence", 0.25))
        if len(zone) < 3:
            return jsonify({"error": "Intrusion zone needs at least 3 points"}), 400
        if not prompt:
            return jsonify({"error": "Open-vocabulary prompt is required (comma-separated labels)"}), 400
        if not beta_ai_mod.owlv2_available():
            return jsonify({"error": "OWLv2 not available. Install transformers pillow torch"}), 400
        zone_tuples = [(p["x"], p["y"]) if isinstance(p, dict) else tuple(p) for p in zone]
        config = {
            "intrusion_zone": [list(t) for t in zone_tuples],
            "prompt": prompt,
            "canvas_width": canvas_w,
            "canvas_height": canvas_h,
            "confidence": conf,
        }
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "intrusion", 1, _dump_cfg(config, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in intrusion_procs:
            intrusion_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule(config, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        try:
            proc = IntrusionDetectionProcessor(
                camera_id, video_readers[camera_id], zone_tuples, prompt, (canvas_w, canvas_h), conf
            )
            intrusion_procs[camera_id] = proc
            proc.start()
        except Exception as e:
            logger.exception("Enable intrusion failed for %s", camera_id)
            intrusion_procs.pop(camera_id, None)
            return jsonify({"error": f"Could not start intrusion detection: {e}"}), 500
        return jsonify({"success": True})

    @app.route("/api/detection/intrusion/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_disable_intrusion(camera_id):
        if camera_id in intrusion_procs:
            intrusion_procs[camera_id].stop()
            del intrusion_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "intrusion"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})

    @app.route("/api/fr/status", methods=["GET"])
    @login_required
    def api_fr_status():
        return jsonify(
            {
                "available": False,
                "active_cameras": list(fr_procs.keys()),
                "stats": {cid: proc.stats for cid, proc in fr_procs.items()},
                "insightface_available": False,
                "yolo_available": False,
                "deepsort_available": False,
                "datasets": 0,
                "defaults": {"threshold": 0.3, "draw_threshold": 0.55, "det_threshold": 0.25},
            }
        )

    @app.route("/api/fr/datasets", methods=["GET"])
    @login_required
    def api_fr_datasets():
        return jsonify({"datasets": []})

    @app.route("/api/fr/folder/add", methods=["POST"])
    @login_required
    def api_fr_add_folder():
        return jsonify({"success": True, "demo": True, "message": "FR dataset management is demo-only in this build"})

    @app.route("/api/fr/scan_and_sync", methods=["POST"])
    @login_required
    def api_fr_scan_and_sync():
        return jsonify({"success": True, "demo": True, "total_synced": 0})

    @app.route("/api/fr/folder/delete", methods=["POST"])
    @login_required
    def api_fr_delete_folder():
        return jsonify({"success": True, "demo": True})

    @app.route("/api/fr/collect/start", methods=["POST"])
    @login_required
    def api_fr_collect_start():
        return jsonify({"success": True, "demo": True, "session_id": "demo_fr_session"})

    @app.route("/api/fr/collect/stop", methods=["POST"])
    @login_required
    def api_fr_collect_stop():
        return jsonify({"success": True, "demo": True})

    @app.route("/api/fr/collect/status/<session_id>", methods=["GET"])
    @login_required
    def api_fr_collect_status(session_id):
        return jsonify(
            {
                "error": None,
                "counts": {},
                "completed": [],
                "person_name": "",
                "csv_updated": False,
                "stopped": True,
                "demo": True,
            }
        )

    @app.route("/api/fr/detection/enable/<camera_id>", methods=["POST"])
    @login_required
    def api_fr_enable(camera_id):
        if not _ensure_camera(camera_id):
            return jsonify({"error": "Camera not found"}), 404
        data = request.json or {}
        config = {
            "threshold": float(data.get("threshold", 0.3)),
            "draw_threshold": float(data.get("draw_threshold", 0.55)),
            "det_threshold": float(data.get("det_threshold", 0.25)),
        }
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO detection_configs (camera_id, detection_type, enabled, config_json) VALUES (?,?,?,?)",
            (camera_id, "fr_detection", 1, _dump_cfg(config, data)),
        )
        conn.commit()
        conn.close()
        if camera_id in fr_procs:
            fr_procs[camera_id].stop()
        if not is_schedule_active(attach_schedule(config, data)):
            return jsonify({"success": True, "scheduled_idle": True})
        proc = DummyDetectionProcessor(camera_id, "fr", config)
        fr_procs[camera_id] = proc
        proc.start()
        return jsonify({"success": True, "demo": True})

    @app.route("/api/fr/detection/disable/<camera_id>", methods=["POST"])
    @login_required
    def api_fr_disable(camera_id):
        if camera_id in fr_procs:
            fr_procs[camera_id].stop()
            del fr_procs[camera_id]
        conn = get_db()
        conn.execute("DELETE FROM detection_configs WHERE camera_id=? AND detection_type=?", (camera_id, "fr_detection"))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
