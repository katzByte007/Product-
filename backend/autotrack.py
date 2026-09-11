"""Shared cross-camera trail for Autotrack."""
import threading
from datetime import datetime

_lock = threading.Lock()
_events = []
_MAX = 24


def reset_trail():
    with _lock:
        _events.clear()


def record_event(camera_id, description):
    desc = (description or "").strip()
    now = datetime.now()
    with _lock:
        if not desc:
            return _info_unlocked(duplicate=True)
        if _events:
            last = _events[-1]
            same = last["camera_id"] == camera_id and last["description"].lower() == desc.lower()
            if same and (now - last["ts"]).total_seconds() < 12:
                return _info_unlocked(duplicate=True)
        is_hop = bool(_events) and _events[-1]["camera_id"] != camera_id
        is_first = not _events
        _events.append({"camera_id": camera_id, "description": desc[:220], "ts": now})
        if len(_events) > _MAX:
            del _events[: len(_events) - _MAX]
        return _info_unlocked(duplicate=False, is_hop=is_hop, is_first=is_first)


def _narrative(events):
    if not events:
        return ""
    first, last = events[0], events[-1]
    t0 = first["ts"].strftime("%H:%M:%S")
    t1 = last["ts"].strftime("%H:%M:%S")
    cams = []
    for ev in events:
        if not cams or cams[-1] != ev["camera_id"]:
            cams.append(ev["camera_id"])
    if len(cams) == 1:
        return f"{first['description']} on {first['camera_id']} at {t0} (continuous)"
    hops = []
    for ev in events:
        stamp = ev["ts"].strftime("%H:%M:%S")
        hops.append(f"{ev['camera_id']} at {stamp} ({ev['description']})")
    return (
        f"{first['description']} tracked across cameras. "
        f"First on {first['camera_id']} at {t0}. "
        f"Track: " + " → ".join(hops) + f". Now on {last['camera_id']} at {t1}."
    )


def _format_unlocked(events):
    if not events:
        return ""
    parts = []
    for ev in events:
        stamp = ev["ts"].strftime("%H:%M:%S")
        parts.append(f"{ev['camera_id']} @ {stamp} — {ev['description']}")
    return "Cross-camera track: " + " → ".join(parts)


def _info_unlocked(duplicate=False, is_hop=False, is_first=False):
    narrative = _narrative(_events)
    return {
        "duplicate": duplicate,
        "is_hop": is_hop,
        "is_first": is_first,
        "cameras": list(dict.fromkeys(ev["camera_id"] for ev in _events)),
        "trail": _format_unlocked(_events),
        "narrative": narrative,
    }


def format_trail():
    with _lock:
        return _format_unlocked(_events)
