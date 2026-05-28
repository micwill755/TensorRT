# Edge LLM Export Benchmarks

This folder contains lightweight export/runtime benchmark utilities for comparing Edge LLM runtime execution against PyTorch eager VLA inference.

## Utilities

- `benchmark_edge_vs_eager.py`: runs the Edge `action_inference` binary and a PyTorch eager adapter, then writes `benchmark_summary.json`.
- `vla_eager_adapter.py`: loads VLA models through the same HF source path used by `tools/edgellm/export_from_hf.py` and dispatches to PI0.5, GR00T, or a generic VLA runner.
- `vla_test_data.py`: abstracts future-trajectory test data for minADE, with JSON, Alpamayo, and LeRobot data sources.

## Notes

The Edge wall-clock time includes process startup, engine loading, plugin loading, runtime setup, and CUDA graph capture. For runtime performance, compare Edge `runtime_e2e_avg` against eager timed latency.

`minADE` is computed by comparing predicted trajectory points against a future ground-truth trajectory. The data source must match the model/task semantics for the accuracy number to be meaningful.

## Captured Results

### GR00T / Alpamayo Ground Truth

Command shape:

```bash
python3 tests/export/benchmark_edge_vs_eager.py \
  --model nvidia/GR00T-N1.5-3B \
  --family vla \
  --model_class gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7 \
  --test_data_source alpamayo \
  --alpamayo_clip_id 030c760c-ae38-49aa-9ad8-f5650a545d26 \
  --alpamayo_t0_us 5100000 \
  --iterations 1 --warmup 0 \
  ...
```

Summary captured on Jetson Thor:

| Metric | Edge | PyTorch eager |
| --- | ---: | ---: |
| Process wall avg | 104553.590 ms | 844.495 ms |
| Runtime E2E avg | 634.784 ms | n/a |
| minADE avg | 16.5881 m | 16.9458 m |

Quality comparison:

| Metric | Value |
| --- | ---: |
| Trajectory mean abs diff | 0.7866 m |
| Trajectory max abs diff | 2.4624 m |
| Trajectory mean L2 diff | 1.2534 m |
| ADE diff | 0.3577 m |
| Quality thresholds passed | false |

Runtime-only speedup, using Edge runtime E2E vs eager latency: about `1.33x`.

### PI0.5 / LeRobot Ground Truth

Command shape:

```bash
python3 tests/export/benchmark_edge_vs_eager.py \
  --model lerobot/pi05_base \
  --family vla \
  --model_class lerobot.policies.pi05.modeling_pi05.PI05Policy \
  --test_data_source lerobot \
  --lerobot_dataset_repo_id lerobot/libero \
  --lerobot_episode_index 0 \
  --lerobot_frame_index 0 \
  --lerobot_future_steps 50 \
  --iterations 1 --warmup 0 \
  ...
```

Summary captured on Jetson Thor:

| Metric | Edge | PyTorch eager |
| --- | ---: | ---: |
| Process wall avg | 95839.462 ms | 844.683 ms |
| Runtime E2E avg | 230.795 ms | n/a |
| minADE avg | 0.2647 m | 0.0233 m |

Quality comparison:

| Metric | Value |
| --- | ---: |
| Trajectory mean abs diff | 0.0511 m |
| Trajectory max abs diff | 0.0522 m |
| Trajectory mean L2 diff | 0.0722 m |
| ADE diff | 0.2415 m |
| Quality thresholds passed | false |

Runtime-only speedup, using Edge runtime E2E vs eager latency: about `3.66x`.
