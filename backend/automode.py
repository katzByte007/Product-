"""Autotrack — OWLv2 object watch across selected cameras."""
import json
import logging

from backend.beta_ai import BetaOwlv2Processor, owlv2_available
from backend.db import get_db
from backend.runtime import automode_procs, video_readers
from backend.schedule_util import is_schedule_active

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "person"


def as_owl_labels(prompt):
    raw = (prompt or "").strip()
    if not raw:
        return DEFAULT_PROMPT
    if raw.lower().startswith("you are") or len(raw) > 180:
        return DEFAULT_PROMPT
    if "," not in raw and len(raw.split()) > 5:
        return DEFAULT_PROMPT
    return raw


class AutoModeOwlv2Processor(BetaOwlv2Processor):
    """Detect prompted objects; alert after continuous hits; stitch cameras."""

    CONTINUOUS_HITS = 2

    def __init__(self, camera_id, camera_reader, prompt_texts, conf=0.2):
        super().__init__(camera_id, camera_reader, prompt_texts, conf)
        self._hit_streak = 0
        self._total_alerts = 0
        self.stats = dict(self.stats)
        self.stats["alerts"] = 0

    def _decorate_frame(self, vis, kept, labels):
        from backend.beta_ai import _draw_hud
        from backend.autotrack import format_trail

        trail = format_trail()
        hud2 = trail[:100] if trail else f"Watching: {', '.join(labels[:6])}"
        _draw_hud(
            vis,
            f"Autotrack (OWLv2)  |  {len(kept)} det  |  Alerts: {self._total_alerts}",
            hud2,
            (20, 120, 180),
        )
        return True

    def _on_detections(self, frame, vis, kept, labels):
        if not kept:
            self._hit_streak = 0
            return
        self._hit_streak += 1
        if self._hit_streak < self.CONTINUOUS_HITS:
            return

        from backend.alerts import save_alert_event
        from backend.autotrack import record_event

        names = []
        seen = set()
        for d in kept:
            lab = (d.get("label") or "object").strip()
            key = lab.lower()
            if key not in seen:
                seen.add(key)
                names.append(lab)
        desc = f"{', '.join(names)} detected"
        info = record_event(self.camera_id, desc)
        if info.get("duplicate"):
            return
        self._total_alerts += 1
        self.stats["alerts"] = self._total_alerts
        hop = bool(info.get("is_hop"))
        message = info.get("narrative") or info.get("trail") or f"{self.camera_id}: {desc}"
        save_alert_event(
            self.camera_id,
            "automode",
            vis if vis is not None else frame,
            [(d.get("box") or (0, 0, 0, 0), d.get("label") or desc, (20, 120, 180)) for d in kept[:6]],
            severity="high",
            meta={"message": message[:500]},
            force=hop,
            cooldown_sec=0 if hop else 10,
        )


def get_automode_config():
    conn = get_db()
    row = conn.execute("SELECT * FROM automode_settings WHERE id=1").fetchone()
    conn.close()
    if not row:
        return {
            "enabled": False,
            "prompt": DEFAULT_PROMPT,
            "camera_ids": [],
            "confidence": 0.2,
            "schedule_enabled": False,
            "schedule_start": "",
            "schedule_end": "",
        }
    try:
        camera_ids = json.loads(row["camera_ids"] or "[]")
    except Exception:
        camera_ids = []
    return {
        "enabled": bool(row["enabled"]),
        "prompt": as_owl_labels((row["prompt"] or "").strip() or DEFAULT_PROMPT),
        "camera_ids": camera_ids,
        "confidence": float(row["confidence"] if row["confidence"] is not None else 0.2),
        "schedule_enabled": bool(row["schedule_enabled"]) if "schedule_enabled" in row.keys() else False,
        "schedule_start": (row["schedule_start"] if "schedule_start" in row.keys() else "") or "",
        "schedule_end": (row["schedule_end"] if "schedule_end" in row.keys() else "") or "",
    }


def save_automode_config(enabled, prompt, camera_ids, confidence, schedule_enabled=False, schedule_start="", schedule_end=""):
    conn = get_db()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(automode_settings)").fetchall()}
    if "schedule_enabled" not in cols:
        conn.execute("ALTER TABLE automode_settings ADD COLUMN schedule_enabled INTEGER DEFAULT 0")
    if "schedule_start" not in cols:
        conn.execute("ALTER TABLE automode_settings ADD COLUMN schedule_start TEXT DEFAULT ''")
    if "schedule_end" not in cols:
        conn.execute("ALTER TABLE automode_settings ADD COLUMN schedule_end TEXT DEFAULT ''")
    prompt = as_owl_labels(prompt)
    conn.execute(
        """
        INSERT INTO automode_settings (id, enabled, prompt, camera_ids, confidence, schedule_enabled, schedule_start, schedule_end)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            enabled=excluded.enabled,
            prompt=excluded.prompt,
            camera_ids=excluded.camera_ids,
            confidence=excluded.confidence,
            schedule_enabled=excluded.schedule_enabled,
            schedule_start=excluded.schedule_start,
            schedule_end=excluded.schedule_end
        """,
        (
            1 if enabled else 0,
            prompt,
            json.dumps(camera_ids),
            float(confidence),
            1 if schedule_enabled else 0,
            schedule_start or "",
            schedule_end or "",
        ),
    )
    conn.commit()
    conn.close()


def stop_automode_processors():
    for cid in list(automode_procs.keys()):
        try:
            automode_procs[cid].stop()
        except Exception:
            pass
        automode_procs.pop(cid, None)


def start_automode_processors(camera_ids, prompt, confidence):
    from backend.autotrack import reset_trail

    stop_automode_processors()
    reset_trail()
    prompt = as_owl_labels(prompt)
    if not prompt:
        return {"started": [], "errors": ["Object prompt is required, e.g. person"]}

    started = []
    errors = []
    for cid in camera_ids:
        if cid not in video_readers:
            errors.append(f"{cid}: camera offline")
            continue
        proc = AutoModeOwlv2Processor(cid, video_readers[cid], [prompt], confidence)
        proc.start()
        if proc.stats.get("status") in ("OWLv2 not available", "Model unavailable", "No labels"):
            errors.append(f"{cid}: {proc.stats.get('status')}")
            proc.stop()
            continue
        automode_procs[cid] = proc
        started.append(cid)
        logger.info("Autotrack OWLv2 started on %s (%s)", cid, prompt)
    return {"started": started, "errors": errors}


def apply_automode(enabled, prompt, camera_ids, confidence, schedule_enabled=False, schedule_start="", schedule_end=""):
    prompt = as_owl_labels(prompt)
    save_automode_config(
        enabled, prompt, camera_ids, confidence, schedule_enabled, schedule_start, schedule_end
    )
    cfg = {
        "schedule_enabled": schedule_enabled,
        "schedule_start": schedule_start,
        "schedule_end": schedule_end,
    }
    if not enabled:
        stop_automode_processors()
        return {"enabled": False, "active_cameras": [], "errors": [], "prompt": prompt}
    if not is_schedule_active(cfg):
        stop_automode_processors()
        return {
            "enabled": True,
            "active_cameras": [],
            "errors": [],
            "scheduled_idle": True,
            "owlv2_available": owlv2_available(),
            "prompt": prompt,
        }
    result = start_automode_processors(camera_ids, prompt, confidence)
    return {
        "enabled": True,
        "active_cameras": result["started"],
        "errors": result["errors"],
        "owlv2_available": owlv2_available(),
        "prompt": prompt,
    }


def restore_automode_from_db():
    cfg = get_automode_config()
    if not cfg["enabled"]:
        return
    if not cfg["camera_ids"]:
        return
    if not is_schedule_active(cfg):
        return
    start_automode_processors(cfg["camera_ids"], cfg["prompt"], cfg["confidence"])
