# VMS Scaling Roadmap and Success Criteria

Target: 50 concurrent camera streams with attribute-level AI detection.
This roadmap supersedes the earlier 100-camera / fixed YOLO + OWLv2 target.

## Target Specification

| Parameter | Target |
| --- | --- |
| Concurrent streams | 50 |
| Per-camera detection load | At least three configured detection outcomes |
| Configuration A | Two fine-tuned custom YOLO models plus Autotrack |
| Configuration B | One prompt-based Beta AI workload covering multiple prompts plus Autotrack |
| Autotrack model | Exactly one deployment-level choice: OWLv2 or YOLO-World |
| Lower-cost comparison | NVIDIA GPU with 8 GB VRAM, 8-12 CPU cores, 32 GB RAM; not an assumed full-workload pass target |
| Probable comfortable starting point | NVIDIA GPU with 24 GB VRAM, 16 CPU cores, 64 GB RAM; provisional until the selected full workload is benchmarked and soaked |
| Operation | 24/7 RTSP streaming, with no process crashes or required restarts |

Three detection outcomes are a business requirement, not a requirement for
three separate expensive model forwards. Multiple prompts in one Beta AI
request must be measured as one logical workload and must retain per-prompt
results for accuracy evaluation.

## Roadmap

| Phase | Work | Status / exit evidence |
| --- | --- | --- |
| 0. Baseline | Establish M4/MPS limits and identify inference serialization versus ingest cost. | Complete: local report records roughly 1 OWLv2 forward/sec at saturation; this is not a 50-camera capacity claim. |
| 1. Scaling PR validation | On Intel Core Ultra 7 / Windows / 16 GB, run six-camera baseline, 30-camera ingest-only, and 30-camera one-detector-per-camera tests. | In progress. A local Apple M4/MPS cloned-file rehearsal is recorded in `data/benchmarks/PHASE1_LOCAL_MPS_REHEARSAL.md`; the required Windows validation-host runs remain outstanding. |
| 2. Model evaluation | Compare OWLv2 and YOLO-World on the same labeled attribute-prompt set. Include precision/recall and latency/throughput under the same hardware and thresholds. | Next. Select one Autotrack model only after accuracy and speed results are reviewed. |
| 3. Target-hardware validation | Run the selected configuration at 50 streams and full three-outcome prompt/model load on the proposed 24 GB VRAM / 16-core / 64 GB starting host; include the 8 GB lower-cost comparison if available. | Planned. Validate resource ceilings and agreed per-camera FPS/latency; adjust GPU count from measured sustained throughput and VRAM headroom. |
| 4. Reliability soak | Run the full target workload continuously for the agreed multi-day duration. | Planned. Require zero crashes, restarts, stream loss, and resource-limit violations. |
| 5. Handover | Deliver code, deployment architecture, sizing assumptions, performance report, and remaining limits. | Planned. |

## Success Gate

All checks must pass simultaneously on target hardware:

- 50 concurrent streams remain healthy without dropped streams or process crashes.
- Every camera receives its selected full detection workload at the FPS and latency agreed after Phase 2.
- A continuous multi-day soak completes with no crash or required restart.
- Attribute-level precision and recall meet the agreed thresholds on a labeled validation set. Model selection cannot trade away the accuracy bar.
- CPU, GPU/VRAM, and RAM remain within the target box limits under full load.
- Autotrack works end-to-end across camera views with the single selected model.
- All handover deliverables are complete, including documented assumptions and remaining bottlenecks.

## Decisions Required Before Final Acceptance

- Agree on the minimum per-camera detection FPS and maximum p95 latency after Phase 2.
- Agree on precision and recall thresholds, dataset size, and validation split.
- Specify the soak-test duration and objective stream-health criteria.
- Confirm whether cross-camera tracking requires persistent identity matching; the current Autotrack trail records detections/hops and is not evidence of identity-level re-identification.
- Confirm whether the three detections mean three named prompts/classes, three simultaneously reported attributes, or three independent event outcomes.

## Benchmark Profiles Added

`benchmark_native.py` now has `yolo-single` (one real YOLO model task per
camera) and `owlv2-multiprompt` (one OWLv2 task with the three configured
labels per forward). Existing profiles remain available for comparison.
Autotrack now persists one OWLv2/YOLO-World selection, shares one YOLO-World
runtime across its selected cameras, and applies the configurable
`VISION_AUTOTRACK_INFER_FPS` budget to either selected model. RTSP readers now
reopen failed captures with bounded backoff and open/read timeouts. These
changes do not establish target-hardware capacity or soak-test results.
The no-AI ingest baseline is measured separately by
`benchmark_comprehensive.py --modes ingest`; its callback profiles are
synthetic and must not be presented as model-throughput measurements.

Phase 2 accuracy evaluation uses JSONL, one image per line. Boxes are pixel
coordinates `[x1, y1, x2, y2]` in the original image; include every prompt on
each record, even when that prompt has no positive annotation, so false
positives are counted:

```json
{"image":"images/frame-001.jpg","prompts":["white helmet","black shirt","blue gloves"],"annotations":[{"label":"white helmet","box":[120,45,210,170]}]}
```

Keep the same camera/sequence split, confidence threshold, IoU threshold, and
prompt list for both models. Split by camera or contiguous sequence to avoid
near-duplicate frames leaking between validation and holdout sets.

```bash
python evaluate_attribute_models.py data/attribute-validation.jsonl --model both --device cuda --confidence 0.25 --iou-threshold 0.5 --output data/benchmarks/attribute-model-evaluation.json
```

The evaluator reports per-prompt and aggregate precision/recall plus warm
steady-state p50/p95 end-to-end latency. It does not choose acceptance
thresholds, and it must be run on the same labeled dataset and target hardware
for the model decision to be valid.

Phase 1 commands, from the repository root:

```bash
python benchmark_comprehensive.py --camera-counts 6,30 --modes ingest --duration 60 --output data/benchmarks/phase1_ingest.csv
python benchmark_native.py --camera-counts 6,30 --modes yolo-single --device cpu --duration 60 --task-interval 0.5 --output data/benchmarks/phase1_single_yolo.csv
```

On a machine with a supported CUDA GPU, use `--device cuda` for the second
command. Record OS, CPU, RAM, GPU, driver, model weight names, video sources,
and the exact command with each result. The benchmark profiles are measurement
tools; they do not establish that the target hardware or product acceptance
criteria have passed.

The 2026-09-30 local MPS rehearsal is partial Phase 1 evidence only. Its
30-camera, 60-second YOLO run completed 93 forwards with 94.7% scheduler drops,
and its 30-reader ingest run reached 6.89 GB RSS on a 16 GB Mac. These local
results are overload indicators, not target NVIDIA capacity or Phase 1 exit
evidence. The hardware figures above are planning estimates, not proven
minimums: first reduce model forwards per camera, then benchmark sustained
per-model throughput with at least 30% reserve on the selected CUDA host.