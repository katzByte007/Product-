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

### AI workload scaling

OWLv2 uses one shared latest-frame scheduler per application process. It polls
active camera readers, keeps at most one pending frame per camera, drops stale
frames under load, and dispatches work fairly to bounded workers. Configure the
worker limit with `VISION_OWLV2_WORKERS` (default `1`; increase only when using
independent model replicas or a device that supports concurrent forwards).
YOLO workloads already use shared model engines with micro-batching.

For an enterprise deployment, run camera groups as separate worker processes
and assign each process to a GPU. Keep the Flask/API process separate from
inference workers. Use bounded process queues, NVMe for alert snapshots, and
64 GB RAM or more for 100 mixed-resolution streams. Do not create one OS
process per camera; inference should be scheduled across cameras.

### Scaling benchmark

Measured on an Apple M4 MacBook Air (10 cores, 16 GB RAM, macOS, MPS) on
2026-09-17 using the local `cam4_industry_ppe.mp4` input and local OWLv2
weights:

| Workload | Result |
| --- | --- |
| 100 scheduler registrations, 2 workers, 5 seconds | 100 cameras observed; 1,995 jobs completed; 2,813 stale jobs dropped; 0 errors |
| One real Beta OWLv2 camera, 16 seconds | 3 completed inferences; annotated frame published; 0 scheduler errors |
| Real Beta + intrusion processors, 20 seconds | Both processors active; both annotated frames published; 292 jobs submitted; 287 stale jobs dropped; 0 scheduler errors |
| Previous per-camera OWLv2 benchmark | 0.73 FPS, 1.37 s p50 latency, with all model calls contending on one mutex |
| Previous 100-reader decode benchmark | 100 active readers, 1.98 GB RSS, 535% CPU before AI inference |

The improvement is primarily a latency and resource-control change, not a
claim that one OWLv2 replica can process 100 cameras at video rate. The old
design created one polling/inference loop per camera and allowed stale work to
wait behind the singleton model. The new design has one dispatcher, one latest
frame slot per camera, bounded workers, fair admission, and explicit drops.
This keeps memory bounded and ensures cameras continue receiving service when
the model is saturated.

The primary Beta, Auto Mode, intrusion, and legacy analytics OWLv2 processors
use this scheduler. Their processor-specific post-processing and alert
contracts remain intact, while frame acquisition and model work are centrally
scheduled. The shared model is still serialized, so additional GPU worker
processes and model replicas are required for high-rate 100-camera inference.

### Cloned-feed comparison

The replay harness `benchmark_cloned_feeds.py` points 100 production
`VideoFileReader` instances at the same local MP4 and registers three task
profiles per camera. On the Apple M4 test host, with a 6-second measurement,
0.2-second task interval, and 4 ms callback workload:

| Metric | Legacy 300 polling threads | Shared bounded scheduler |
| --- | ---: | ---: |
| Active readers | 100 | 100 |
| Logical tasks | 300 | 300 |
| Callback throughput | 1,715/sec | 819/sec |
| Callback p50 | 5.14 ms | 5.11 ms |
| RSS | 2.271 GB | 1.877 GB |
| Process CPU time | 23.72 sec | 22.03 sec |
| Errors | Not bounded by scheduler | 0 |
| Stale-frame handling | No explicit bound | 5,183 dropped submissions |

This control test does not show a raw callback-FPS increase because the old
design allowed 300 callbacks to run concurrently without an admission limit.
The improvement is bounded resource usage, explicit overload behavior, and
predictable latest-frame latency. Real model throughput must be measured with
the target GPU and model replicas; the earlier local OWLv2 result remains
approximately 0.73 FPS per serialized replica at 1.37 s p50 latency.

An additional real-model replay used three cloned Beta OWLv2 processors on the
same feed for 24 seconds. It completed 6 total OWLv2 inferences, or 0.25
aggregate FPS, dropped 576 stale submissions, and reported 0 scheduler errors.
This is lower than the isolated one-camera result because all three workloads
share one serialized in-process model. It confirms the scheduler protects the
application under saturation; it does not prove throughput scaling. Throughput
scaling requires running the new `GpuWorkerProcess` replicas on separate GPUs
and connecting their result queues to the application control plane.

The scheduler now rotates its polling cursor across registered jobs, applies
the same `VISION_OWLV2_WORKERS` default regardless of which processor starts
first, namespaces jobs by analytics type, and exposes status through the
existing `/api/health` and `/api/system-stats` endpoints. Health polling does
not create inference workers when no OWLv2 workload is active.

The batch/process interfaces are available for enterprise deployment in
`backend/inference_protocol.py` and `backend/gpu_worker.py`. YOLO batches use
the existing `InferenceEngine` micro-batch path. OWLv2 requests are grouped by
model and identical label tuple, so only compatible label sets share a forward
batch. `GpuWorkerProcess` provides a bounded inter-process request queue and a
GPU-pinned worker process. It is opt-in so the one-command demo remains simple.

Run the synthetic 100-camera and three-task harness with:

```bash
python benchmark_scaling.py --duration 5 --workers 2 --interval 0.1
```

Run the process lifecycle smoke test with:

```bash
python benchmark_gpu_worker.py
```

---

## What works after setup

- Camera video streams from `data/videos/`
- OWLv2 after `python download_owlv2.py` (or HuggingFace download on first use)
- YOLO models from `models/`
- Windows → Linux path migration for stored camera paths
