"""Download google/owlv2-base-patch16-ensemble into this project.

The HuggingFace cache folder (models--google--owlv2-base-patch16-ensemble) is too
large for GitHub. Run this once after cloning:

    python download_owlv2.py

Weights are written to models/owlv2/ which the app auto-discovers.
Override with VISION_OWLV2_LOCAL_PATH if you keep the model somewhere else.
"""
from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(ROOT, "models", "owlv2")
REPO_ID = os.environ.get("VISION_OWLV2_MODEL", "google/owlv2-base-patch16-ensemble")


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is missing. Install deps first:", flush=True)
        print("  python -m pip install -r requirements.txt", flush=True)
        return 1

    os.makedirs(DEST, exist_ok=True)
    print(f"Downloading {REPO_ID}", flush=True)
    print(f"Destination: {DEST}", flush=True)
    print("This is ~600 MB and can take several minutes.", flush=True)
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=DEST,
        )
    except Exception as exc:
        print(f"Download failed: {exc}", flush=True)
        print("Check your internet connection and retry.", flush=True)
        return 1

    config = os.path.join(DEST, "config.json")
    if not os.path.isfile(config):
        print("Download finished but config.json was not found in the destination.", flush=True)
        return 1

    print("OWLv2 weights are ready.", flush=True)
    print(f"Local path: {DEST}", flush=True)
    print("Start the app with:  python app.py", flush=True)
    print(
        "To use a different folder, set VISION_OWLV2_LOCAL_PATH before starting the app.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
