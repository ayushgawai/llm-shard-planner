import argparse, json, sys
from .config import load_model, load_device, ModelConfig
from .planner import enumerate_plans
from .cost import layer_cost

def main(argv=None):
    p = argparse.ArgumentParser(prog="llm-shard-planner")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan",):
        s = sub.add_parser(name); s.add_argument("--model", required=True); s.add_argument("--device", required=True)
        s.add_argument("--devices", type=int, default=8); s.add_argument("--seq", type=int, default=1024)
        s.add_argument("--batch", type=int, default=1); s.add_argument("--top", type=int, default=5)
    s = sub.add_parser("validate"); s.add_argument("--world", type=int, default=2); s.add_argument("--moe", action="store_true"); s.add_argument("--profile", action="store_true")
    s = sub.add_parser("bench"); s.add_argument("--world", type=int, default=2)
    a = p.parse_args(argv)

    if a.cmd == "plan":
        cfg, dev = load_model(a.model), load_device(a.device)
        feas, infeas = enumerate_plans(cfg, dev, devices=a.devices, seq=a.seq, batch=a.batch)
        c = layer_cost(cfg, dev, seq=a.seq, batch=a.batch)
        print(f"model={cfg.name} layers={cfg.layers} moe={cfg.is_moe} per-layer: {c.flops/1e9:.2f} GFLOP, {c.hbm_bytes/1e6:.1f} MB HBM, KV@{a.seq}={c.kv_bytes/1e9:.2f} GB")
        print(f"{'tp':>3} {'ep':>3} {'pp':>3} {'GB/dev':>8} {'prefill_s':>10} {'decode_tok/s':>13}")
        for pl in feas[:a.top]:
            print(f"{pl.tp:>3} {pl.ep:>3} {pl.pp:>3} {pl.mem_per_device_gb:>8} {pl.prefill_latency_s:>10.4f} {pl.decode_tok_s:>13.1f}")
        print(f"rejected: {len(infeas)} plan(s) over memory/device constraints")
        for pl in infeas[:3]: print(f"  tp={pl.tp} ep={pl.ep} pp={pl.pp}: {pl.reason}")
        return 0 if feas else 1
    if a.cmd == "validate":
        from .sharded import validate
        cfg = ModelConfig(name="tiny-moe", experts=4, top_k=2) if a.moe else ModelConfig(name="tiny-dense")
        res = validate(cfg, world_size=a.world, profile=a.profile)
        print(json.dumps(res, indent=2))
        return 0 if res["max_abs_error"] < 1e-4 else 1
    if a.cmd == "bench":
        from .fused import bench as fbench
        from .sharded import validate
        print("fused RMSNorm+residual:", json.dumps(fbench()))
        res = validate(ModelConfig(), world_size=a.world, profile=True)
        print("sharded vs unsharded:", json.dumps(res, indent=2))
        return 0 if res["max_abs_error"] < 1e-4 else 1

if __name__ == "__main__": sys.exit(main())
