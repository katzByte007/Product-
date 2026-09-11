# Food Industry Vision AI

Vision AI app for camera monitoring (PPE, intrusion, workforce, open-vocabulary OWLv2, and more).

Default login: **admin** / **admin**  
Default port: **8765**

---

## 1. Run the application

```bash
git clone https://github.com/katzByte007/Product-.git
cd Product-
python -m pip install -r requirements.txt
python download_owlv2.py
python app.py
```

**Windows (PowerShell)** from this folder (`D:\platform` or your clone):

```powershell
cd D:\platform
python -m pip install -r requirements.txt
python download_owlv2.py
python app.py
```

Open the URL printed in the terminal, e.g. `http://127.0.0.1:8765`.

The UI is served from `frontend/dist`. After changing React/JSX/CSS:

```bash
cd frontend
npm install
npm run build
```

Then restart `python app.py`.

---

## 2. Install OWLv2 (not in GitHub)

`google/owlv2-base-patch16-ensemble` is too large to store in this repo. Install it locally once:

```bash
python download_owlv2.py
```

That downloads HuggingFace model **google/owlv2-base-patch16-ensemble** into `models/owlv2/`. The app auto-discovers that folder.

Equivalent one-liner (same destination):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/owlv2-base-patch16-ensemble', local_dir='models/owlv2')"
```

Or with the HuggingFace CLI:

```bash
huggingface-cli download google/owlv2-base-patch16-ensemble --local-dir models/owlv2
```

If you already have a HuggingFace cache folder such as `models--google--owlv2-base-patch16-ensemble`, point the app at the snapshot that contains `config.json` **and** weight files (`.safetensors` / `.bin`):

```powershell
$env:VISION_OWLV2_LOCAL_PATH = "D:\platform\models--google--owlv2-base-patch16-ensemble\snapshots\<revision>"
python app.py
```

Without local weights, OWLv2 will try to download from HuggingFace on first use.

---

## 3. Change paths for your machine

Paths default to folders next to `app.py`. Override them with environment variables **before** starting the app.

| What | Environment variable | Default |
| --- | --- | --- |
| OWLv2 weights | `VISION_OWLV2_LOCAL_PATH` | `models/owlv2` (auto-discovered) |
| HuggingFace model id | `VISION_OWLV2_MODEL` | `google/owlv2-base-patch16-ensemble` |
| Camera videos | `VISION_VIDEOS_DIR` | `data/videos` |
| YOLO / detector weights | — | `models\` (e.g. `yolov8m.pt`, `ANPRlib.pt`) |
| Port | `FOOD_VISION_PORT` | `8765` |
| Bind host | `VISION_HOST` | `0.0.0.0` |
| Public hostname in logs | `VISION_PUBLIC_HOST` | machine hostname / LAN IP |
| GPU / CPU | `VISION_VLM_DEVICE` | `auto` (`cuda` if a GPU is present) |

**Windows (PowerShell)**

```powershell
$env:VISION_OWLV2_LOCAL_PATH = "D:\models\owlv2"
$env:VISION_VIDEOS_DIR = "D:\videos"
$env:FOOD_VISION_PORT = "8765"
$env:VISION_VLM_DEVICE = "cuda"
python app.py
```

**Linux / macOS**

```bash
export VISION_OWLV2_LOCAL_PATH="/opt/models/owlv2"
export VISION_VIDEOS_DIR="/opt/videos"
export FOOD_VISION_PORT=8765
export VISION_VLM_DEVICE=cuda
python app.py
```

Camera `video_path` values stored in the database are remapped automatically when you copy the project to another machine: the app looks up the same filename under the current `VISION_VIDEOS_DIR`.

YOLO weights stay in `models/`. Put files such as `yolov8m.pt` and `pharmappenew.pt` there, or keep the names the app already resolves from `config.py`.

Optional extras: `VISION_PUBLIC_HOST`, `VISION_PORT` (ignored when set to `8080`).

---

## What works after setup

- Camera video streams from `data/videos/`
- OWLv2 after `python download_owlv2.py` (or HuggingFace download on first use)
- YOLO models from `models/`
- Windows → Linux path migration for stored camera paths
