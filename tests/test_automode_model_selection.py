import json
import sqlite3
import types

import backend.automode as automode


def test_automode_model_choice_migrates_and_persists(tmp_path, monkeypatch):
    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE automode_settings (id INTEGER PRIMARY KEY, enabled INTEGER, prompt TEXT, camera_ids TEXT, confidence REAL)"
    )
    connection.execute(
        "INSERT INTO automode_settings VALUES (1, 0, 'person', '[]', 0.2)"
    )
    connection.commit()
    connection.close()

    def get_test_db():
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(automode, "get_db", get_test_db)
    automode.save_automode_config(True, "white helmet, blue gloves", ["cam-1"], 0.3, model="yolo-world")

    config = automode.get_automode_config()
    assert config["model"] == "yolo-world"
    assert config["prompt"] == "white helmet, blue gloves"
    assert config["camera_ids"] == ["cam-1"]

    conn = get_test_db()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(automode_settings)")}
    assert "model" in columns
    assert conn.execute("SELECT model FROM automode_settings WHERE id=1").fetchone()[0] == "yolo-world"
    conn.close()


def test_start_uses_only_selected_autotrack_processor(monkeypatch):
    selected = []
    prepared = []

    class FakeProcessor:
        def __init__(self, camera_id, reader, prompts, confidence):
            selected.append((camera_id, prompts, confidence))
            self.stats = {"status": "Initializing"}

        def start(self):
            self.stats["status"] = "Active"

    monkeypatch.setattr(automode, "stop_automode_processors", lambda: None)
    monkeypatch.setattr(automode, "video_readers", {"cam-1": object()})
    monkeypatch.setattr(automode, "automode_procs", {})
    monkeypatch.setattr(automode, "yolo_world_available", lambda: True)
    monkeypatch.setattr(automode, "AutoModeYoloWorldProcessor", FakeProcessor)
    monkeypatch.setattr(
        automode,
        "get_yolo_world_runner",
        lambda: types.SimpleNamespace(prepare=lambda labels: prepared.append(labels)),
    )

    result = automode.start_automode_processors(
        ["cam-1"], "white helmet, blue gloves", 0.35, "yolo-world"
    )

    assert result == {"started": ["cam-1"], "errors": []}
    assert selected == [("cam-1", ["white helmet, blue gloves"], 0.35)]
    assert prepared == [["white helmet", "blue gloves"]]


def test_unsupported_model_is_rejected():
    try:
        automode.start_automode_processors([], "person", 0.2, "both")
    except ValueError as error:
        assert "owlv2, yolo-world" in str(error)
    else:
        raise AssertionError("unsupported Autotrack model was accepted")