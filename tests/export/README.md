# Edge Export Benchmarks

This folder contains end-to-end export/runtime benchmark entrypoints.  The
first benchmark compares an Edge LLM runtime run against a PyTorch eager
baseline for VLA models such as PI0.5 and GR00T.

## Full VLA Flow

`run_vla_e2e.py` runs the complete sequence:

1. export language, visual, and action engines one role at a time with `tools.edgellm.export_from_hf`
2. run the Edge runtime VLA smoke test through `action_inference`
3. run the Edge-vs-eager benchmark

```bash
python3 tests/export/run_vla_e2e.py \
  --model lerobot/pi05_base \
  --input_file /workspace/artifacts/vlm-exports/pi05_edge_full_vla/action_smoke.json \
  --output_root /workspace/artifacts/vlm-exports/pi05_edge_full_vla \
  --edge_binary /workspace/artifacts/edgellm-build-thor/examples/multimodal/action_inference \
  --export_arg=--model_class=lerobot.policies.pi05.modeling_pi05.PI05Policy \
  --export_arg=--tokenizer=google/paligemma-3b-pt-224 \
  --export_arg=--processor_model=google/paligemma-3b-pt-224 \
  --eager_adapter tests.export.pi05_eager_adapter:create_runner \
  --iterations 10 \
  --warmup 2
```

Use `--print_only` first if you want to inspect the generated export, smoke,
and benchmark commands without running them. The export phase is sequential by
default: `language`, then `visual`, then `action`. Use `--export_role` to run a
subset.

## Benchmark Only

Run it from the Torch-TensorRT repo root:

```bash
python3 tests/export/benchmark_edge_vs_eager.py \
  --model lerobot/pi05_base \
  --model_class lerobot.policies.pi05.modeling_pi05.PI05Policy \
  --input_file /workspace/artifacts/vlm-exports/pi05_edge_full_vla/action_smoke.json \
  --output_dir /workspace/artifacts/benchmarks/pi05_edge_vs_eager \
  --edge_binary /workspace/artifacts/edgellm-build-thor/examples/multimodal/action_inference \
  --engine_dir /workspace/artifacts/llm-exports/pi05_edge_llm \
  --multimodal_engine_dir /workspace/artifacts/vlm-exports/pi05_edge_full_vla \
  --eager_adapter tests.export.pi05_eager_adapter:create_runner \
  --iterations 10 \
  --warmup 2
```
