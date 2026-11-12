# llm-shard-planner

A sharding planner and analytical cost model for distributed LLM inference on
multi-device accelerators, validated against a reference sharded PyTorch
execution (tensor + expert parallelism, `torch.distributed`/gloo, CPU).

## Design

```
  configs/*.json                device_*.json
  (model: dense / MoE)          (HBM, BW, FLOPs, link)
        |                             |
        v                             v
  +------------------------------------------------+
  | cost.py  per-layer: FLOPs, HBM bytes, KV cache,|
  |          collectives (all-reduce / all-gather /|
  |          all-to-all) via alpha-beta link model |
  +------------------------+-----------------------+
                           v
  +------------------------------------------------+
  | planner.py  enumerate TP x EP x PP degrees,    |
  | reject plans over per-device HBM, rank by      |
  | predicted prefill latency, then decode tok/s   |
  +------------------------+-----------------------+
                           v  (chosen degrees)
  +------------------------------------------------+
  | sharded.py  reference execution: column-/row-  |
  | parallel linears, head-sharded attention,      |
  | expert-parallel MoE, all-reduce combine (gloo) |
  +------------------------+-----------------------+
                           v
  validate: max |sharded - unsharded|   profile.py: torch.profiler
                                        compute vs communication
```

| Component | File | What it does |
|---|---|---|
| Configs | `llm_shard_planner/config.py`, `configs/` | Dense (Llama-style) and MoE (GLM/Kimi-style, top-k) model JSON; device profile JSON |
| Cost model | `cost.py` | Per-layer FLOPs, HBM bytes, KV-cache bytes, collective bytes/time (`alpha + volume/link_bw`, ring factors) |
| Planner | `planner.py` | Enumerates TP ∈ {1,2,4,8} × EP × PP, rejects over-memory plans, ranks the rest |
| Reference model | `model.py` | Small unsharded decoder (GQA attention, SwiGLU or top-k MoE) — ground truth |
| Sharded execution | `sharded.py` | Column/row-parallel linears, sharded heads, expert-sliced MoE; spawned gloo ranks (up to 8) |
| Fused kernel | `fused.py` | RMSNorm + residual: Triton kernel on CUDA, single-expression PyTorch fallback on CPU |
| Profiler | `profile.py` | `torch.profiler` on one sharded decode step, split into compute vs communication time |

## Run

```bash
python3 -m venv .venv && .venv/bin/python -m pip install torch pytest
.venv/bin/python -m pip install -e .

# Plan: rank TP/EP/PP for a model on a device profile
.venv/bin/python -m llm_shard_planner.cli plan --model configs/llama_7b.json --device configs/device_80gb.json --devices 8 --seq 1024
.venv/bin/python -m llm_shard_planner.cli plan --model configs/kimi_moe.json --device configs/device_24gb.json --devices 8 --seq 1024

# Validate: sharded (2 gloo CPU ranks) vs unsharded, prints max abs error
.venv/bin/python -m llm_shard_planner.cli validate --world 2
.venv/bin/python -m llm_shard_planner.cli validate --world 2 --moe --profile

# Bench: fused RMSNorm microbench + sharded-vs-unsharded + profiler summary
.venv/bin/python -m llm_shard_planner.cli bench
```

## Test

```bash
.venv/bin/python -m pytest tests -q
# 8 passed
```

Covers: cost-model sanity (KV growth, alpha-beta collectives, top-k MoE FLOPs),
memory rejection, planner ranking/feasibility, fused RMSNorm correctness, and
sharded-vs-unsharded equivalence for dense and MoE models (world size 2).

## Laptop sanity runs (real output, Apple-silicon laptop, CPU/gloo)

These are deliberately tiny execution configs — the planner numbers for the
7B-scale config are analytical predictions, not measured throughput.

`plan` (dense, 7B-scale config, 8× 80GB profile):

```
model=llama-7b-scale layers=32 moe=False per-layer: 431.64 GFLOP, 421.5 MB HBM, KV@1024=0.54 GB
 tp  ep  pp   GB/dev  prefill_s  decode_tok/s
  8   1   1    2.219     0.0102         477.8
  4   1   1    3.938     0.0148         442.1
  2   1   1    7.376     0.0251         280.9
rejected: 0 plan(s) over memory/device constraints
```

`validate --world 2 --moe --profile` (tiny MoE, one sharded decode step profiled):

```
"max_abs_error": 1.19e-07
"profile": {"compute_us": 800.7, "comm_us": 70.6, "comm_pct": 8.1,
  "top_ops": [{"op": "aten::mm", "self_cpu_us": 137.8},
              {"op": "aten::_scaled_dot_product_flash_attention_for_cpu", "self_cpu_us": 126.4},
              {"op": "c10d::allreduce_", "self_cpu_us": 70.6}]}
```

`bench` (fused RMSNorm+residual, CPU fallback — no GPU on this laptop, so the
PyTorch fallback is *not* faster than the naive version here; the Triton path
only runs on CUDA):

```
fused RMSNorm+residual: {"naive_ms": 0.0276, "fused_ms": 0.0297, "backend": "pytorch"}
sharded vs unsharded: {"world_size": 2, "max_abs_error": 8.94e-08, ...}
```

## Limitations (honest)

- **Analytical, not measured.** Planner latencies are roofline estimates
  (`max(compute, HBM) + comm`); no kernel efficiency, overlap, or scheduling
  effects. Treat rankings as relative, not absolute.
- **Pipeline parallelism is memory-only in the model.** PP is enumerated and
  splits weights/KV across stages, but it does not reduce single-request
  latency in this model (no microbatch throughput modelling), and the
  reference execution does not implement pipeline stages.
- **MoE dispatch is not a true all-to-all.** The cost model charges all-to-all
  for MoE, but the reference execution uses a replicated router + all-reduce
  combine over expert-sliced weights. Outputs are identical; the wire pattern
  differs from a production expert-parallel runtime.
- **Execution is CPU/gloo at tiny scale.** Sharded validation runs small
  configs (`hidden=128`, 4 layers) on up to 8 CPU processes. No CUDA, no
  NCCL, no KV-cache reuse in execution (each forward recomputes attention),
  no quantization, no continuous batching.
- **Memory model is approximate:** weights + KV + a flat 0.5 GB
  activation/workspace allowance per device. Real runtimes fragment and
  reserve more.
- The Triton fused kernel is untested on this machine (no GPU); only the
  PyTorch fallback path is exercised by tests here.

## License

MIT — see `LICENSE`.
