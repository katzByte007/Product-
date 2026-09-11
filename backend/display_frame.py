"""Build composite display frames (raw + AI overlays) for live streaming."""
import cv2

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


def _draw_hud(frame, name, proc):
    if name == "headcount":
        s = proc.stats or {}
        cv2.putText(
            frame,
            f"Heads: {s.get('current_count', 0)}  |  Total: {s.get('total_entries', 0)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    elif name == "entryexit":
        s = proc.stats if hasattr(proc, "stats") else {}
        cv2.putText(
            frame,
            f"Entries: {s.get('entries', 0)}  Exits: {s.get('exits', 0)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Inside: {s.get('current_inside', 0)}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    elif name == "workforce":
        s = proc.stats if hasattr(proc, "stats") else {}
        color = (0, 0, 255) if s.get("alert") else (0, 255, 0)
        cv2.putText(
            frame,
            f"Worker {s.get('worker_id', '-')}  |  Dwell: {s.get('dwell_sec', 0)}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    elif name == "automode":
        s = proc.stats if hasattr(proc, "stats") else {}
        cv2.putText(
            frame,
            f"Autotrack  |  Alerts: {s.get('alerts', 0)}  |  {s.get('status', '')}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 120, 180),
            2,
        )


def build_display_frame(camera_id: str):
    from backend.analytics_runtime import _draw_premium_ppe_box

    active = []
    if camera_id in headcount_procs:
        active.append(("headcount", headcount_procs[camera_id]))
    if camera_id in entryexit_procs:
        active.append(("entryexit", entryexit_procs[camera_id]))
    if camera_id in flapgate_procs:
        active.append(("flapgate", flapgate_procs[camera_id]))
    if camera_id in workforce_procs:
        active.append(("workforce", workforce_procs[camera_id]))
    if camera_id in intrusion_procs:
        active.append(("intrusion", intrusion_procs[camera_id]))
    if camera_id in ppe_procs:
        active.append(("ppe", ppe_procs[camera_id]))
    if camera_id in fire_smoke_procs:
        active.append(("fire_smoke", fire_smoke_procs[camera_id]))
    if camera_id in anpr_procs:
        active.append(("anpr", anpr_procs[camera_id]))
    if camera_id in beta_procs:
        active.append(("beta", beta_procs[camera_id]))
    if camera_id in automode_procs:
        active.append(("automode", automode_procs[camera_id]))
    if camera_id in fr_procs:
        active.append(("fr", fr_procs[camera_id]))

    reader = video_readers.get(camera_id)
    frame = reader.get_frame() if reader else None
    if frame is None:
        return None
    if not active:
        return frame

    if len(active) == 1:
        _, proc = active[0]
        get_ann = getattr(proc, "get_annotated_frame", None)
        if callable(get_ann):
            ann = get_ann()
            if ann is not None:
                return ann

    frame = frame.copy()
    skip_draw = set()
    for name, proc in active:
        if name in ("workforce", "intrusion", "automode"):
            get_ann = getattr(proc, "get_annotated_frame", None)
            if callable(get_ann):
                ann = get_ann()
                if ann is not None:
                    frame = ann
                    skip_draw.add(name)
                    break

    for name, proc in active:
        if name in skip_draw:
            continue
        draw_zones = getattr(proc, "_draw_zones", None)
        if callable(draw_zones):
            try:
                draw_zones(frame)
            except Exception:
                pass
        get_dets = getattr(proc, "get_last_detections", None)
        if callable(get_dets):
            for bbox, label, color in get_dets():
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                if name in ("ppe", "beta", "automode", "fire_smoke", "anpr", "fr", "intrusion", "workforce"):
                    _draw_premium_ppe_box(frame, x1, y1, x2, y2, label, color)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_y = max(y1 - 6, 14)
                    cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        _draw_hud(frame, name, proc)
    return frame
