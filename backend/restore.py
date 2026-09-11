"""Restore enabled detection processors from SQLite after server boot."""
import json
import logging
import time

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

YOLO_TYPES = ("headcount", "entryexit", "flapgate", "ppe", "firesmoke", "workforce")


def _proc_map():
    return {
        "headcount": headcount_procs,
        "entryexit": entryexit_procs,
        "flapgate": flapgate_procs,
        "ppe": ppe_procs,
        "firesmoke": fire_smoke_procs,
        "workforce": workforce_procs,
        "intrusion": intrusion_procs,
        "anpr": anpr_procs,
        "fr_detection": fr_procs,
    }


def is_detection_running(dtype, cid):
    return cid in _proc_map().get(dtype, {})


def stop_detection_instance(dtype, cid):
    procs = _proc_map().get(dtype)
    if not procs or cid not in procs:
        return
    try:
        procs[cid].stop()
    except Exception:
        pass
    procs.pop(cid, None)


def _load_analytics():
    try:
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
    except Exception as e:
        logger.error("Analytics runtime unavailable: %s", e)
        return None


def restore_detections_from_db():
    time.sleep(1.5)
    conn = get_db()
    configs = conn.execute("SELECT * FROM detection_configs WHERE enabled=1").fetchall()
    conn.close()

    if not configs:
        return

    from backend.schedule_util import is_schedule_active

    needs_yolo = any(cfg["detection_type"] in YOLO_TYPES for cfg in configs)
    A = _load_analytics() if needs_yolo else None

    for cfg in configs:
        cid = cfg["camera_id"]
        dtype = cfg["detection_type"]
        config = json.loads(cfg["config_json"]) if cfg["config_json"] else {}
        if not is_schedule_active(config):
            continue
        start_detection_instance(dtype, cid, config, A=A)


def start_detection_instance(dtype, cid, config, A=None):
    if cid not in video_readers:
        logger.warning("Skipping start %s/%s — camera offline", cid, dtype)
        return
    if is_detection_running(dtype, cid):
        return
    if dtype in YOLO_TYPES and A is None:
        A = _load_analytics()
        if A is None:
            return
    try:
        if dtype == "headcount":
            if not A:
                return
            conf = float(config.get("confidence", 0.1))
            engine = A["get_inference_engine"]("headcount", HEADCOUNT_MODEL_PATH)
            proc = A["HeadCountProcessor"](cid, video_readers[cid], engine, conf)
            headcount_procs[cid] = proc
            proc.start()
        elif dtype == "entryexit":
            if not A:
                return
            entry_z = [tuple(p) for p in config.get("entry_zone", [])]
            exit_z = [tuple(p) for p in config.get("exit_zone", [])]
            if len(entry_z) < 3 or len(exit_z) < 3:
                return
            cw = int(config.get("canvas_width", 800))
            ch = int(config.get("canvas_height", 450))
            conf = float(config.get("confidence", 0.5))
            engine = A["get_inference_engine"]("entryexit", ENTRYEXIT_MODEL_PATH, classes=[0])
            proc = A["EntryExitProcessor"](cid, video_readers[cid], entry_z, exit_z, (cw, ch), engine, conf)
            entryexit_procs[cid] = proc
            proc.start()
        elif dtype == "flapgate":
            if not A:
                return
            gate_zones_raw = config.get("gate_zones", {})
            gate_zones = {int(k): [tuple(p) for p in v] for k, v in gate_zones_raw.items()}
            if len(gate_zones) < 3:
                return
            cw = int(config.get("canvas_width", 800))
            ch = int(config.get("canvas_height", 450))
            conf = float(config.get("confidence", 0.65))
            engine = A["get_inference_engine"]("flapgate", FLAPGATE_MODEL_PATH, classes=[0])
            proc = A["FlapGateProcessor"](cid, video_readers[cid], gate_zones, (cw, ch), engine, conf)
            flapgate_procs[cid] = proc
            proc.start()
        elif dtype == "ppe":
            if not A:
                return
            conf = float(config.get("confidence", 0.4))
            engine = A["get_inference_engine"]("ppe", PPE_MODEL_PATH, classes=None)
            proc = A["PPEDetectionProcessor"](cid, video_readers[cid], engine, conf)
            ppe_procs[cid] = proc
            proc.start()
        elif dtype == "firesmoke":
            if not A:
                return
            conf = float(config.get("confidence", 0.35))
            engine = A["get_inference_engine"]("firesmoke", FIRE_SMOKE_MODEL_PATH, classes=None)
            proc = A["FireSmokeDetectionProcessor"](cid, video_readers[cid], engine, conf)
            fire_smoke_procs[cid] = proc
            proc.start()
        elif dtype == "workforce":
            if not A:
                return
            from backend.workforce_processor import WorkforceMonitoringProcessor

            mz = []
            for p in config.get("machine_zone", []):
                if isinstance(p, dict):
                    mz.append((p["x"], p["y"]))
                else:
                    mz.append((p[0], p[1]))
            if len(mz) < 3:
                return
            cw = int(config.get("canvas_width", 800))
            ch = int(config.get("canvas_height", 450))
            conf = float(config.get("confidence", 0.25))
            away_alert_sec = float(config.get("away_alert_sec", 3))
            engine = A["get_inference_engine"]("workforce", WORKFORCE_MODEL_PATH, classes=[0])
            proc = WorkforceMonitoringProcessor(
                cid, video_readers[cid], mz, (cw, ch), engine, conf, away_alert_sec=away_alert_sec
            )
            workforce_procs[cid] = proc
            proc.start()
        elif dtype == "intrusion":
            from backend.intrusion_processor import IntrusionDetectionProcessor

            zone = []
            for p in config.get("intrusion_zone", []):
                if isinstance(p, dict):
                    zone.append((p["x"], p["y"]))
                else:
                    zone.append((p[0], p[1]))
            prompt = (config.get("prompt") or "").strip()
            if len(zone) < 3 or not prompt:
                return
            cw = int(config.get("canvas_width", 800))
            ch = int(config.get("canvas_height", 450))
            conf = float(config.get("confidence", 0.25))
            proc = IntrusionDetectionProcessor(cid, video_readers[cid], zone, prompt, (cw, ch), conf)
            intrusion_procs[cid] = proc
            proc.start()
        elif dtype == "anpr":
            proc = DummyDetectionProcessor(cid, "anpr", config)
            anpr_procs[cid] = proc
            proc.start()
        elif dtype == "fr_detection":
            proc = DummyDetectionProcessor(cid, "fr", config)
            fr_procs[cid] = proc
            proc.start()
        logger.info("Restored %s for %s", dtype, cid)
    except Exception as e:
        logger.error("Restore %s/%s failed: %s", cid, dtype, e, exc_info=True)


def restore_beta_from_db(start_beta_fn):
    """Restore beta processors using existing beta save logic."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM beta_detection_prompts ORDER BY id").fetchall()
    backend = "owlv2"
    confidence = 0.35
    try:
        s = conn.execute("SELECT model, confidence FROM beta_settings WHERE id=1").fetchone()
        if s:
            backend = s["model"] or "owlv2"
            if s["confidence"] is not None:
                confidence = float(s["confidence"])
    except Exception:
        pass
    conn.close()

    from backend.schedule_util import is_schedule_active
    sched = {}
    try:
        conn = get_db()
        srow = conn.execute("SELECT * FROM beta_settings WHERE id=1").fetchone()
        conn.close()
        if srow:
            keys = srow.keys()
            sched = {
                "schedule_enabled": bool(srow["schedule_enabled"]) if "schedule_enabled" in keys else False,
                "schedule_start": srow["schedule_start"] if "schedule_start" in keys else "",
                "schedule_end": srow["schedule_end"] if "schedule_end" in keys else "",
            }
    except Exception:
        sched = {}
    if not is_schedule_active(sched):
        return

    camera_to_prompts = {}
    for r in rows:
        if not r["enabled"]:
            continue
        pt = (r["prompt_text"] or "").strip()
        if not pt:
            continue
        try:
            cids = json.loads(r["camera_ids"]) if r["camera_ids"] else []
        except Exception:
            cids = []
        for cid in cids:
            camera_to_prompts.setdefault(cid, []).append(pt)

    if camera_to_prompts:
        start_beta_fn(camera_to_prompts, confidence, backend)
