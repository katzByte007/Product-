"""Active-hours windows for Auto Mode and AI feature detections."""
from datetime import datetime


def _norm_hhmm(value):
    text = str(value or "").strip()
    if not text:
        return ""
    parts = text.replace(".", ":").split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        return ""
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


def attach_schedule(config, data):
    out = dict(config or {})
    if data is None:
        return out
    if "schedule_enabled" in data:
        out["schedule_enabled"] = bool(data.get("schedule_enabled"))
    if "schedule_start" in data:
        out["schedule_start"] = _norm_hhmm(data.get("schedule_start"))
    if "schedule_end" in data:
        out["schedule_end"] = _norm_hhmm(data.get("schedule_end"))
    return out


def is_schedule_active(cfg, now=None):
    """True when detection should run. Missing/disabled schedule = always on."""
    if not cfg:
        return True
    if not cfg.get("schedule_enabled"):
        return True
    start = _norm_hhmm(cfg.get("schedule_start"))
    end = _norm_hhmm(cfg.get("schedule_end"))
    if not start or not end:
        return True
    now = now or datetime.now()
    current = now.hour * 60 + now.minute
    start_m = int(start[:2]) * 60 + int(start[3:5])
    end_m = int(end[:2]) * 60 + int(end[3:5])
    if start_m == end_m:
        return True
    if start_m < end_m:
        return start_m <= current < end_m
    return current >= start_m or current < end_m
