# Comprehensive Cloned-Feed Benchmark

Date: 2026-09-25

Machine: Apple Silicon MacBook Air, macOS 27.0, Python 3.14.7

Command:

```text
python3 benchmark_comprehensive.py --camera-counts 3,6,12,25,50,75,100 --modes ingest,fixed,owlv2,full --duration 5 --workers 2 --task-interval 0.2 --output data/benchmarks/comprehensive_cloned_feed_benchmark.csv
```

## Test design

The benchmark replayed three local MP4 feeds as 3, 6, 12, 25, 50, 75, and 100 virtual cameras. It recorded ingest throughput, scheduler submissions and completions, bounded-queue drops, callback latency, frame age, RSS, and CPU time for four modes:

- `ingest`: video-reader capacity only.
- `fixed`: two fixed-model tasks per camera.
- `owlv2`: three OWLv2 tasks per camera.
- `full`: two fixed-model tasks plus three OWLv2 tasks per camera.

The benchmark uses deterministic callback work durations from `MODEL_WORK_MS` to measure scheduler capacity. It is therefore an application scheduling and backpressure benchmark, not a measurement of real YOLO or OWLv2 model accuracy or native GPU kernel latency.

## Results

| Cameras | Ingest read FPS | Fixed completion | Fixed frame-age p95 | OWLv2 completion | OWLv2 drops | Full completion | Full drops |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 118.4 | 1.0000 | 41.4 ms | 0.9015 | 19 | 0.8270 | 69 |
| 6 | 220.9 | 1.0000 | 40.8 ms | 0.4539 | 273 | 0.4090 | 497 |
| 12 | 380.3 | 1.0000 | 41.0 ms | 0.2332 | 753 | 0.2105 | 1,299 |
| 25 | 144.9 | 0.9922 | 59.8 ms | 0.3119 | 463 | 0.4098 | 431 |
| 50 | 94.8 | 0.9856 | 96.8 ms | 0.2950 | 529 | 0.4768 | 293 |
| 75 | 105.5 | 1.0000 | 62.0 ms | 0.3352 | 420 | 0.4544 | 353 |
| 100 | 99.7 | 1.0000 | 106.1 ms | 0.3216 | 430 | 0.5021 | 268 |

### Capacity findings

- The fixed two-task workload remained effectively stable at 100 cameras: 533 submitted, 533 completed, zero drops, and zero errors in the final run. Its worst measured frame-age p95 was 106.1 ms.
- OWLv2 saturated at the two-worker limit. Completion fell from 90.15% at 3 cameras to 45.39% at 6 cameras and remained approximately 23% to 34% through 100 cameras, with 228 to 753 completions per run and substantial latest-frame replacement drops.
- Full mode also saturated. Completion was 82.70% at 3 cameras and approximately 41% to 50% from 6 to 100 cameras. No scheduler errors were recorded.
- OWLv2 callback p95 latency was approximately 57 ms at low load and 62 ms at 100 cameras. The main failure signal is queue pressure and dropped stale work, rather than callback execution latency alone.
- Ingest memory reached approximately 5.9 GB at 100 cameras. Ingest throughput became irregular above 12 cameras, including the OpenCV/FFmpeg stream timeout warnings emitted during the run.

## Stability and optimization baseline

The earlier 100-camera scheduler stress test completed 14,700 of 14,700 submitted jobs with zero drops and zero errors. That test measured scheduler throughput with a lightweight callback and should be interpreted separately from this model-weighted benchmark.

The motion-gating simulation reduced modeled inference jobs from 90,000 to 2,827, a 96.86% reduction. This explains why sparse OWLv2 execution is necessary, but this benchmark shows that three continuously scheduled OWLv2 tasks per camera still exceed a two-worker capacity envelope.

## Conclusion

On this machine and configuration, 100-camera ingest and the fixed-model scheduler are stable, but continuous three-task OWLv2 service is not capacity-safe with only two workers. The next production improvement should be workload reduction at the OWLv2 admission layer: motion gating, heartbeat scheduling, ROI cropping, and per-camera fairness. A native-model benchmark with actual YOLO and OWLv2 inference should be run separately before making hardware-capacity claims.