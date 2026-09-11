# Food Industry Vision AI

## Run (Linux server)

```bash
cd "~/aicoe-workspace/computer-vision/food industry"
python app.py
```

Open the URL printed in the terminal, e.g. `http://spark-f22e:8765` — login: **admin** / **admin**

Default port is **8765** (8080 is reserved for Utthunga and is ignored even if `VISION_PORT=8080` is set in your shell).

All paths (videos, models, database) are resolved automatically from the project folder. No `.env`, npm, or shell scripts needed.

**First time only** on a new machine: `pip install -r requirements.txt`

---

### What works out of the box
- 6 camera video streams from `data/videos/`
- OWLv2 / Qwen models auto-discovered in project folder
- Windows → Linux path migration for stored camera paths

### Optional env vars (only if you need them)
`VISION_PORT`, `VISION_PUBLIC_HOST`, `VISION_VLM_DEVICE=cuda`

### Beta AI deps (optional)
`pip install torch transformers accelerate safetensors`
