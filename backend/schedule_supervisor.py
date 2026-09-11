"""Start/stop detections according to configured active-hours windows."""
import json
import logging
import threading
import time

from backend.db import get_db
from backend.runtime import automode_procs, beta_procs
from backend.schedule_util import is_schedule_active

logger = logging.getLogger(__name__)


def _beta_schedule_cfg():
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM beta_settings WHERE id=1").fetchone()
    except Exception:
        row = None
    conn.close()
    if not row:
        return {}
    keys = row.keys()
    return {
        "schedule_enabled": bool(row["schedule_enabled"]) if "schedule_enabled" in keys else False,
        "schedule_start": row["schedule_start"] if "schedule_start" in keys else "",
        "schedule_end": row["schedule_end"] if "schedule_end" in keys else "",
    }


def _stop_all_beta():
    for cid in list(beta_procs.keys()):
        try:
            beta_procs[cid].stop()
        except Exception:
            pass
        beta_procs.pop(cid, None)


def enforce_schedules(start_beta_fn=None):
    from backend.restore import (
        is_detection_running,
        restore_beta_from_db,
        start_detection_instance,
        stop_detection_instance,
    )

    conn = get_db()
    rows = conn.execute("SELECT * FROM detection_configs WHERE enabled=1").fetchall()
    conn.close()
    for row in rows:
        cid = row["camera_id"]
        dtype = row["detection_type"]
        try:
            config = json.loads(row["config_json"]) if row["config_json"] else {}
        except Exception:
            config = {}
        due = is_schedule_active(config)
        running = is_detection_running(dtype, cid)
        if running and not due:
            stop_detection_instance(dtype, cid)
            logger.info("Schedule pause %s/%s", dtype, cid)
        elif due and not running:
            start_detection_instance(dtype, cid, config)

    try:
        from backend.automode import get_automode_config, start_automode_processors, stop_automode_processors

        cfg = get_automode_config()
        due = bool(cfg.get("enabled")) and is_schedule_active(cfg)
        running = bool(automode_procs)
        if running and not due:
            stop_automode_processors()
            logger.info("Schedule pause Auto Mode")
        elif due and not running and cfg.get("camera_ids"):
            start_automode_processors(cfg["camera_ids"], cfg["prompt"], cfg["confidence"])
    except Exception as e:
        logger.debug("Auto Mode schedule: %s", e)

    if start_beta_fn:
        try:
            beta_due = is_schedule_active(_beta_schedule_cfg())
            if not beta_due and beta_procs:
                _stop_all_beta()
                logger.info("Schedule pause Beta")
            elif beta_due and not beta_procs:
                restore_beta_from_db(start_beta_fn)
        except Exception as e:
            logger.debug("Beta schedule: %s", e)


def start_schedule_supervisor(start_beta_fn=None):
    def _loop():
        time.sleep(8)
        while True:
            try:
                enforce_schedules(start_beta_fn=start_beta_fn)
            except Exception as e:
                logger.error("Schedule supervisor: %s", e)
            time.sleep(20)

    threading.Thread(target=_loop, daemon=True, name="ScheduleSupervisor").start()
