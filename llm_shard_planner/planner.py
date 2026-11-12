from dataclasses import dataclass
from .cost import layer_cost, weights_bytes

@dataclass
class Plan:
    tp: int; ep: int; pp: int
    mem_per_device_gb: float
    prefill_latency_s: float
    decode_tok_s: float
    feasible: bool = True
    reason: str = ""

def _divisors_ok(cfg, tp, ep):
    return (cfg.heads % tp == 0 and cfg.kv_heads % tp == 0
            and (not cfg.is_moe or (cfg.experts % ep == 0 and ep >= 1))
            and (not cfg.is_moe or cfg.expert_intermediate % 1 == 0))

def enumerate_plans(cfg, dev, *, devices=8, seq=1024, batch=1, context=None):
    context = context or seq
    plans = []
    cands = [d for d in (1,2,4,8) if d <= devices]
    for tp in cands:
        for pp in cands:
            for ep in (cands if cfg.is_moe else [1]):
                if tp*pp > devices or max(tp, ep)*pp > devices: continue
                if cfg.layers % pp or not _divisors_ok(cfg, tp, ep): continue
                plans.append(_eval(cfg, dev, tp, ep, pp, seq, batch, context))
    feas = sorted([p for p in plans if p.feasible],
                  key=lambda p: (p.prefill_latency_s, -p.decode_tok_s))
    return feas, [p for p in plans if not p.feasible]

def _eval(cfg, dev, tp, ep, pp, seq, batch, context):
    stages = cfg.layers // pp
    # weights: attn+dense sharded by TP, experts by EP, layers by PP
    w = weights_bytes(cfg)
    if cfg.is_moe:
        # ponytail: approximate split - non-expert weights by TP, expert bulk by EP
        w_dev = w * 0.1/(tp*pp) + w * 0.9/(ep*pp)
    else:
        w_dev = w / (tp*pp)
    kv_dev = layer_cost(cfg, dev, seq=1, batch=batch, tp=tp, ep=ep, context=context).kv_bytes / (tp*pp)
    mem = (w_dev + kv_dev) / 1e9 + 0.5  # +0.5GB activation/workspace fudge
    cap = dev.hbm_gb
    feas = mem <= cap
    pre = layer_cost(cfg, dev, seq=seq, batch=batch, tp=tp, ep=ep, context=seq)
    dec = layer_cost(cfg, dev, seq=1, batch=batch, tp=tp, ep=ep, context=context)
    # PP bubble: (pp-1)/(pp) rough penalty on prefill only
    bubble = 1 + (pp-1)/max(seq,1) if pp > 1 else 1.0
    # PP splits memory but does NOT cut single-request latency (stages run in
    # sequence for one request); it only adds a small bubble. TP/EP cut per-layer work.
    return Plan(tp, ep, pp, round(mem,3),
                  pre.layer_time_s*cfg.layers*bubble,
                  1.0/max(dec.layer_time_s*cfg.layers, 1e-12),
                  feas, "" if feas else f"needs {mem:.1f}GB > {cap}GB HBM")
