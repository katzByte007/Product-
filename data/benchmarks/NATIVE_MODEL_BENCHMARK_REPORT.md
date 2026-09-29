# Native Model Benchmark

Date: 2026-09-25

Machine: Apple Silicon MacBook Air, macOS 27.0, Python 3.14.7, PyTorch 2.14.0, Apple MPS

The benchmark uses real local YOLO and OWLv2 weights, three cloned MP4 feeds, a latest-frame scheduler, and one native worker. Each run lasts five seconds with a 0.5 second admission interval. Rows are appended incrementally to `native_model_benchmark.csv`.

Command used for the completed scaling runs:

```text
python3 benchmark_native.py --camera-counts 3,6,12,25,50,75,100 --modes yolo,owlv2,mixed,baseline,optimized --duration 5 --workers 1 --task-interval 0.5 --device mps --output data/benchmarks/native_model_benchmark.csv
```

## Instrumentation

Each row records actual model forwards/sec, requested and completed forwards, detections, alerts, drop rate, p50/p95/p99 end-to-end latency, frame age, queue wait, preprocessing, model-forward and postprocessing time, CPU, RSS, per-camera completion, per-camera frame age, and starvation count. MPS does not expose the CUDA/NVML telemetry used for GPU utilization, GPU temperature, or dedicated GPU memory, so those fields are recorded as unavailable rather than estimated.

## Verified results

| Mode | Cameras | Actual forwards | Completed | Drop rate | Latency p95 | Frame age p95 | Model p50 | RSS | Starved cameras |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO | 3 | 48 | 48 | 20.0% | 551 ms | 289 ms | 57 ms | 1.91 GB | 0 |
| YOLO | 6 | 103 | 103 | 14.2% | 519 ms | 473 ms | 57 ms | 2.58 GB | 0 |
| YOLO | 12 | 98 | 98 | 59.2% | 1,149 ms | 1,109 ms | 61 ms | 3.01 GB | 0 |
| YOLO | 25 | 3 | 3 | 87.0% | 3,833 ms | 1,750 ms | 3,021 ms | 3.52 GB | 23 |
| YOLO | 50 | 1 | 1 | 75.0% | 11,532 ms | 1 ms | 11,532 ms | 3.71 GB | 49 |
| YOLO | 75 | 2 | 2 | 82.7% | 9,734 ms | 1,356 ms | 5,399 ms | 4.07 GB | 74 |
| YOLO | 100 | 2 | 2 | 85.0% | 9,273 ms | 749 ms | 4,550 ms | 4.17 GB | 99 |
| OWLv2 | 3 | 11 | 11 | 81.1% | 1,935 ms | 1,360 ms | 540 ms | 2.95 GB | 0 |
| OWLv2 | 6 | 11 | 11 | 85.6% | 2,068 ms | 1,482 ms | 556 ms | 3.42 GB | 2 |
| OWLv2 | 12 | 9 | 9 | 88.1% | 1,468 ms | 786 ms | 605 ms | 2.81 GB | 9 |
| OWLv2 | 25 | 9 | 9 | 73.3% | 4,818 ms | 3,855 ms | 713 ms | 3.03 GB | 22 |
| OWLv2 | 50 | 6 | 6 | 69.2% | 3,370 ms | 2,173 ms | 651 ms | 4.25 GB | 48 |
| Baseline OWLv2 | 3 | 8 | 8 | 83.3% | 1,489 ms | 794 ms | 711 ms | 1.56 GB | 0 |
| Baseline OWLv2 | 6 | 9 | 9 | 86.1% | 1,959 ms | 1,224 ms | 659 ms | 2.24 GB | 3 |
| Baseline OWLv2 | 12 | 7 | 7 | 87.0% | 1,703 ms | 709 ms | 665 ms | 2.99 GB | 9 |
| Optimized OWLv2 | 3 | 9 | 9 | 82.2% | 1,738 ms | 1,051 ms | 656 ms | 2.93 GB | 0 |
| Optimized OWLv2 | 6 | 8 | 8 | 86.1% | 1,491 ms | 732 ms | 679 ms | 3.05 GB | 3 |
| Optimized OWLv2 | 12 | 7 | 7 | 87.1% | 2,248 ms | 1,487 ms | 657 ms | 3.28 GB | 9 |
| Mixed | 3 | 13 | 13 | 83.3% | 1,061 ms | 704 ms | 566 ms | 2.24 GB | 0 |

All 19 recorded rows completed with zero native model errors.

## Optimization comparison

At 3 cameras, optimized OWLv2 reduced preprocessing p50 from approximately 20.0 ms to 11.2 ms and reduced end-to-end latency p50 from approximately 1,093 ms to 847 ms. It also produced one gated skip. At 6 and 12 cameras, the single native worker remained saturated; motion gating did not materially improve the drop rate because the cloned feeds contained enough frame change to keep admission busy. The optimization is therefore useful for preprocessing and freshness, but it is not a substitute for sparse admission, more capacity, or fewer logical OWLv2 tasks.

## Capacity conclusions

- Real YOLO capacity is approximately 17 forwards/sec at 6 cameras in this run. Saturation becomes obvious at 12 cameras, where drops reach 59.2% and p95 latency exceeds one second.
- Real OWLv2 capacity is approximately 1 forward/sec on this MPS configuration. At 25 cameras, 22 of 25 cameras were starved; at 50 cameras, 48 of 50 were starved.
- The mixed 3-camera workload completed only 13 real forwards in five seconds with 83.3% drops, demonstrating that combining two YOLO tasks and three OWLv2 tasks is dominated by the OWLv2 model.
- MPS unified-memory pressure becomes material at high camera counts. A fresh 75-camera OWLv2 run reached roughly 9 GB RSS and sustained more than seven CPU cores without reaching a stable result-write boundary, so the 75/100 OWLv2, mixed, baseline, and optimized high-scale cases were stopped rather than reported as valid measurements.
- The missing high-scale rows are a measured host limitation, not synthetic zeros. The benchmark retains the partial CSV and supports resumption with `--append` after reducing camera-reader memory or moving inference to a dedicated worker process.

This is a historical native model-capacity result for the former 100-camera
target, not a 50-camera product capacity claim. The revised target and phases
are documented in `VMS_SCALING_ROADMAP.md`. New `yolo-single` and
`owlv2-multiprompt` benchmark profiles represent one YOLO task per camera and
one OWLv2 forward with three labels; those profiles have not yet been run on
the Windows validation host or the target NVIDIA hardware. The M4 data still
supports sparse OWLv2 admission, per-camera fairness, and bounded inference,
but does not establish any refined success criterion.
