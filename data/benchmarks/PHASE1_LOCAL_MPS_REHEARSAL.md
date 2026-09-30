# Phase 1 Local MPS Rehearsal

Date: 2026-09-30

## Purpose and scope

This is the closest live-model rehearsal available in the current workspace,
not a target-hardware acceptance test. It replays local MP4 files as cloned
camera readers on an Apple Silicon MacBook Air (macOS 27.0, 10 CPU cores, 16 GB
RAM, PyTorch 2.14.0, Apple MPS; no CUDA GPU). It does not use RTSP streams,
the Windows Intel Core Ultra 7 validation host, or the target NVIDIA 8 GB GPU.

Inputs are the three local `cam4_industry_ppe.mp4`, `cam5_pharma_ppe.mp4`, and
`cam6_lab_ppe.mp4` clips, repeated to reach the requested virtual camera count.
The native profile uses `models/best10xemphead.pt` for the single YOLO task and
local `models/owlv2/` weights for one three-label OWLv2 forward. One MPS worker
was configured, with a requested task interval of 0.5 seconds. The 10-second
native 30/50-camera runs are overload probes; only the 6/30-camera YOLO runs
were extended to 60 seconds.

Commands:

```text
python3 benchmark_comprehensive.py --camera-counts 6,30 --modes ingest --duration 60 --output data/benchmarks/phase1_local_ingest_mps.csv
python3 benchmark_native.py --camera-counts 6,30 --modes yolo-single --duration 60 --workers 1 --task-interval 0.5 --device mps --output data/benchmarks/phase1_local_yolo_mps_60s.csv
python3 benchmark_native.py --camera-counts 6,30,50 --modes yolo-single,owlv2-multiprompt --duration 10 --workers 1 --task-interval 0.5 --device mps --output data/benchmarks/phase1_local_native_mps.csv
```

Raw CSVs are retained beside this report. The ingest CSV reports `read_fps` as
the benchmark's `get_frame()` sampling rate, not the decoder's independently
measured frame rate. Health ratio is the fraction of sampled reader health
checks that reported a fresh active reader. A transient unhealthy sample does
not necessarily mean a persistent dropped stream; this benchmark does not
measure RTSP stream continuity.

## Results

### Ingest-only, 60 seconds

| Cloned readers | Sampled reads/sec | Frame age p50 / p95 | Healthy samples | Healthy at end | RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 217.5 total | 21 / 42 ms | 100.0% | 6 / 6 | 1.70 GB |
| 30 | 71.4 total | 28 / 279 ms | 95.0% | 30 / 30 | 6.89 GB |

Thirty cloned readers substantially increased memory and CPU use on this host;
the result is not evidence that 50 live RTSP streams fit the 32 GB target box.

### Real YOLO, one detector task per camera, 60 seconds

| Cloned cameras | Completed forwards | Scheduler completion | Drops | Forward rate | End-to-end p95 | Model p50 | Starved cameras | RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 704 | 97.8% | 16 / 720 | 11.54/sec total | 380 ms | 63 ms | 0 | 2.40 GB |
| 30 | 93 | 4.1% | 2,150 / 2,271 | 1.52/sec total | 1,813 ms | 614 ms | 0* | 3.26 GB |

`0*` means each camera received at least one completion during the run, not that
the requested cadence was sustained. At 30 cameras the observed average was
about 0.052 forwards per camera per second, far below the configured 2/sec
request rate.

### Real native-model overload probes, 10 seconds

| Mode | Cameras | Prompts per OWLv2 forward | Completed | Completion | Drops | End-to-end p95 | Starved cameras | RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO single | 6 | n/a | 100 | 83.3% | 16.7% | 371 ms | 0 | 2.34 GB |
| YOLO single | 30 | n/a | 18 | 4.1% | 90.2% | 2,403 ms | 13 | 3.14 GB |
| YOLO single | 50 | n/a | 17 | 3.9% | 89.8% | 2,415 ms | 33 | 3.59 GB |
| OWLv2 multiprompt | 6 | 3 | 18 | 15.0% | 81.7% | 1,196 ms | 0 | 3.65 GB |
| OWLv2 multiprompt | 30 | 3 | 11 | 3.0% | 90.7% | 2,761 ms | 19 | 3.13 GB |
| OWLv2 multiprompt | 50 | 3 | 13 | 3.3% | 90.8% | 1,325 ms | 37 | 3.87 GB |

All six short native runs reported zero model errors. High drop rates and
starvation at 30/50 cameras show scheduler saturation, not successful service.
MPS does not provide a directly comparable 8 GB CUDA VRAM measurement; GPU
utilization and dedicated GPU memory are unavailable in these CSVs.

## Interpretation

- The 6-camera, 60-second local YOLO profile shows the benchmark can run a real
  detector task on cloned feeds with high completion on this MPS host.
- At 30 cameras, the same profile fails the requested task cadence by a wide
  margin. The result identifies saturation in the current one-worker profile;
  it does not predict how the target NVIDIA worker configuration will perform.
- Three-prompt OWLv2 uses one forward per logical camera task, but one MPS
  worker saturates quickly as camera count rises.
- Synthetic scheduler benchmarks and these cloned MP4 runs do not test the
  target process topology, real RTSP reconnect behavior, 24/7 reliability,
  precision/recall, or 8 GB VRAM limits.
- The current app's full workload and both configurations have not been run
  together at 50 cameras. These results cannot pass the project success gate.

## Phase status and next tests

Phase 0 remains complete. Phase 1 has a local MPS rehearsal, but remains in
progress because its required Windows validation-host runs are outstanding.
Phase 2 still needs the shared labeled attribute set, agreed accuracy bar, and
same-hardware OWLv2 versus YOLO-World evaluation. Phase 3 requires the selected
full workload on the target NVIDIA box; Phase 4 requires the agreed multi-day
soak. No acceptance threshold has been inferred from this rehearsal.
