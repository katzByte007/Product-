"""Live + stored analytics for every AI detection type."""
import json
import logging
import threading
import time

from backend.db import get_db
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
    workforce_procs,
)

logger = logging.getLogger(__name__)

_persist_lock = threading.Lock()
_last_persist = 0.0
PERSIST_INTERVAL_SEC = 30.0

LIVE_REGISTRIES = (
    ("headcount", headcount_procs),
    ("entryexit", entryexit_procs),
    ("flapgate", flapgate_procs),
    ("workforce", workforce_procs),
    ("intrusion", intrusion_procs),
    ("ppe", ppe_procs),
    ("firesmoke", fire_smoke_procs),
    ("anpr", anpr_procs),
    ("fr", fr_procs),
    ("beta", beta_procs),
    ("automode", automode_procs),
)

EVENT_QUERIES = {
    "headcount": "SELECT COUNT(*) AS n, COALESCE(SUM(current_count),0) AS current_sum, COALESCE(SUM(total_entries),0) AS entries FROM headcounts",
    "entryexit": "SELECT COUNT(*) AS n, COALESCE(SUM(entries),0) AS entries, COALESCE(SUM(exits),0) AS exits, COALESCE(SUM(current_inside),0) AS inside FROM entryexit_counts",
    "flapgate": "SELECT COUNT(*) AS n, COALESCE(SUM(trespassing_count),0) AS trespassing FROM flapgate_events",
    "ppe": "SELECT COUNT(*) AS n, COALESCE(SUM(compliant_count),0) AS compliant, COALESCE(SUM(violation_count),0) AS violations FROM ppe_events",
    "firesmoke": "SELECT COUNT(*) AS n, COALESCE(SUM(fire_count),0) AS fire, COALESCE(SUM(smoke_count),0) AS smoke FROM fire_smoke_events",
    "anpr": "SELECT COUNT(*) AS n, COALESCE(SUM(is_overspeeding),0) AS overspeeding FROM anpr_events",
}


def _jsonish(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonish(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonish(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (str, int, float)):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _proc_stats(registry):
    out = {}
    for cid, proc in registry.items():
        st = getattr(proc, "stats", None) or {}
        try:
            out[cid] = _jsonish(dict(st))
        except Exception:
            out[cid] = {"status": str(st)}
    return out


def live_processor_stats():
    return {name: _proc_stats(reg) for name, reg in LIVE_REGISTRIES}


def _table_exists(conn, name):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return bool(row)


def stored_event_totals(conn=None):
    close = False
    if conn is None:
        conn = get_db()
        close = True
    totals = {}
    for key, sql in EVENT_QUERIES.items():
        table = sql.split(" FROM ")[-1].strip().split()[0]
        try:
            if not _table_exists(conn, table):
                totals[key] = {"rows": 0}
                continue
            row = conn.execute(sql).fetchone()
            totals[key] = dict(row) if row else {"n": 0}
        except Exception as e:
            logger.debug("event totals %s: %s", key, e)
            totals[key] = {"rows": 0}
    try:
        rows = conn.execute(
            "SELECT detection_type, COUNT(*) AS n FROM alerts GROUP BY detection_type"
        ).fetchall()
        totals["alerts_by_type"] = {r["detection_type"]: r["n"] for r in rows}
        totals["alerts_total"] = sum(totals["alerts_by_type"].values())
    except Exception:
        totals["alerts_by_type"] = {}
        totals["alerts_total"] = 0
    if close:
        conn.close()
    return totals


def persist_live_snapshots(live=None):
    global _last_persist
    now = time.time()
    with _persist_lock:
        if now - _last_persist < PERSIST_INTERVAL_SEC:
            return
        _last_persist = now
        live = live if live is not None else live_processor_stats()
        conn = get_db()
        try:
            for dtype, cams in live.items():
                for cid, st in (cams or {}).items():
                    conn.execute(
                        "INSERT INTO detection_snapshots (camera_id, detection_type, stats_json) VALUES (?,?,?)",
                        (cid, dtype, json.dumps(st, default=str)),
                    )
            conn.execute(
                "DELETE FROM detection_snapshots WHERE id NOT IN (SELECT id FROM detection_snapshots ORDER BY id DESC LIMIT 4000)"
            )
            conn.commit()
        except Exception as e:
            logger.debug("snapshot persist failed: %s", e)
        finally:
            conn.close()


def recent_snapshots(limit=80):
    conn = get_db()
    rows = []
    try:
        if _table_exists(conn, "detection_snapshots"):
            rows = conn.execute(
                "SELECT camera_id, detection_type, stats_json, created_at FROM detection_snapshots ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    except Exception:
        rows = []
    conn.close()
    out = []
    for r in rows:
        try:
            stats = json.loads(r["stats_json"] or "{}")
        except Exception:
            stats = {}
        out.append(
            {
                "camera_id": r["camera_id"],
                "detection_type": r["detection_type"],
                "stats": stats,
                "created_at": r["created_at"],
            }
        )
    return out


def dashboard_payload():
    live = live_processor_stats()
    persist_live_snapshots(live)
    conn = get_db()
    stored = stored_event_totals(conn)
    conn.close()
    active = {name: len(reg) for name, reg in LIVE_REGISTRIES}
    return {
        **live,
        "active_counts": active,
        "stored": stored,
        "history": recent_snapshots(60),
    }
