import time

import app as flask_app
import backend.db as db
from backend.cross_camera_tracking import CameraTransition, CrossCameraTracker, TrackObservation


def _observation(camera_id, track_id, timestamp, zone):
    return TrackObservation(
        camera_id=camera_id,
        local_track_id=track_id,
        timestamp=timestamp,
        frame_id=int(timestamp),
        bbox=(10, 20, 60, 130),
        zone_id=zone,
    )


def test_candidate_confirmation_persists_global_track_segments(tmp_path, monkeypatch):
    database_path = tmp_path / "cross-camera.sqlite"
    monkeypatch.setattr(db, "DB_PATH", str(database_path))
    monkeypatch.setattr(db, "DATA_DIR", str(tmp_path))
    db.init_db()
    db.ensure_demo_user()
    connection = db.get_db()
    connection.executemany(
        "INSERT INTO cameras (camera_id, name, video_path) VALUES (?,?,?)",
        [("cam-a", "Camera A", "a.mp4"), ("cam-b", "Camera B", "b.mp4")],
    )
    connection.commit()
    connection.close()

    tracker = CrossCameraTracker([
        CameraTransition("cam-a", "cam-b", 2, 30, "exit", "entry")
    ], tracklet_gap_sec=1)
    monkeypatch.setattr(flask_app, "cross_camera_tracker", tracker)
    started = time.time()
    tracker.observe_camera(
        "cam-a", "headcount", [_observation("cam-a", 17, started, "exit")], now=started
    )
    tracker.observe_camera("cam-a", "headcount", [], now=started + 2)
    candidates = tracker.observe_camera(
        "cam-b", "headcount", [_observation("cam-b", 42, started + 8, "entry")],
        now=started + 8,
    )
    candidate_id = candidates[0].candidate_global_id

    client = flask_app.app.test_client()
    assert client.post("/api/login", json={"username": "admin", "password": "admin"}).status_code == 200
    response = client.post(
        f"/api/cross-camera/candidates/{candidate_id}/confirm",
        json={"global_track_id": "P123"},
    )
    assert response.status_code == 200
    assert response.json["global_track_id"] == "P123"
    assert client.post(f"/api/cross-camera/candidates/{candidate_id}/confirm").status_code == 409

    connection = db.get_db()
    global_track = connection.execute(
        "SELECT global_track_id, current_camera_id, status FROM global_tracks"
    ).fetchone()
    segments = connection.execute(
        "SELECT camera_id, local_track_id FROM global_track_segments ORDER BY camera_id"
    ).fetchall()
    connection.close()
    assert tuple(global_track) == ("P123", "cam-b", "active")
    assert [tuple(segment) for segment in segments] == [("cam-a", "17"), ("cam-b", "42")]

    listed = client.get("/api/cross-camera/global-tracks")
    assert listed.status_code == 200
    assert listed.json[0]["global_track_id"] == "P123"
    assert len(listed.json[0]["segments"]) == 2

    connection = db.get_db()
    connection.execute(
        "UPDATE global_tracks SET last_seen=? WHERE global_track_id='P123'",
        (time.time() - flask_app.cross_camera_tracker.identity_timeout_sec - 1,),
    )
    connection.commit()
    connection.close()
    assert client.get("/api/cross-camera/global-tracks").json[0]["status"] == "temporarily_lost"

    connection = db.get_db()
    connection.execute(
        "UPDATE global_tracks SET last_seen=? WHERE global_track_id='P123'",
        (time.time() - flask_app.cross_camera_tracker.identity_timeout_sec * 2 - 1,),
    )
    connection.commit()
    connection.close()
    assert client.get("/api/cross-camera/global-tracks").json[0]["status"] == "closed"