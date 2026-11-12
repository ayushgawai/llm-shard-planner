"""Analytical per-layer cost model. Roofline: time = max(compute, HBM) + comm."""
from dataclasses import dataclass

@dataclass
class Cost:
    flops: float = 0.0
    hbm_bytes: float = 0.0
    kv_bytes: float = 0.0      # total KV cache (all layers, whole model)
    comm_bytes: float = 0.0
    comm_time_s: float = 0.0
    layer_time_s: float = 0.0

def collective_time(nbytes, kind, degree, dev):
    """Alpha-beta ring / all-to-all approximation."""
    if degree <= 1 or nbytes == 0: return 0.0
    alpha = dev.link_latency_us * 1e-6
    bw = dev.link_bw_gbs * 1e9
    if kind == "all_reduce":   msgs, vol = degree, 2*(degree-1)/degree*nbytes
    elif kind == "all_gather": msgs, vol = degree, (degree-1)/degree*nbytes
    elif kind == "all_to_all": msgs, vol = degree, (degree-1)/degree*nbytes
    else: raise ValueError(kind)
    return msgs*alpha + vol/bw

def layer_cost(cfg, dev, *, seq, batch=1, tp=1, ep=1, context=None):
    ctx = context if context is not None else seq
    B, S, H, d = batch, seq, cfg.hidden, cfg.head_dim
    q, kv = cfg.heads*d, cfg.kv_heads*d
    db = cfg.dtype_bytes
    flops = hbm = comm_bytes = comm_t = 0.0
    # attention projections + scores + output
    flops += 2*B*S*H*(q+2*kv) + 2*B*S*q*H
    flops += 2*2*B*cfg.heads*S*ctx*d  # QK^T and AV
    w_attn = (H*(q+2*kv) + q*H) * db / tp
    # MLP
    if cfg.is_moe:
        active = cfg.top_k
        flops += 2*B*S*H*cfg.experts  # router
        flops += active * 3 * 2*B*S*H*cfg.expert_intermediate
        w_mlp = cfg.experts * 3*H*cfg.expert_intermediate * db / max(ep,1)
        # all-to-all dispatch + combine: tokens x hidden, twice
        cb = 2 * B*S*H*db
        comm_bytes += cb; comm_t += collective_time(cb, "all_to_all", max(ep,1), dev)
    else:
        flops += 3 * 2*B*S*H*cfg.intermediate
        w_mlp = 3*H*cfg.intermediate * db / tp
    hbm += w_attn + w_mlp + 2*B*S*H*db  # activations (rough)
    if tp > 1:  # row-parallel outputs: attn + MLP all-reduce
        cb = 2 * B*S*H*db
        comm_bytes += cb; comm_t += collective_time(cb, "all_reduce", tp, dev)
    shard = max(ep, 1) if cfg.is_moe else tp
    compute_t = flops / (dev.peak_tflops*1e12 * shard)  # per-device FLOPs = total/shard
    hbm_t = hbm / (dev.hbm_bw_gbs*1e9)
    kv_total = 2 * cfg.layers * B * ctx * cfg.kv_heads * d * db
    return Cost(flops, hbm, kv_total, comm_bytes, comm_t, max(compute_t, hbm_t) + comm_t)

def weights_bytes(cfg):
    H, d = cfg.hidden, cfg.head_dim
    per_layer_attn = H*(cfg.heads*d + 2*cfg.kv_heads*d) + (cfg.heads*d)*H
    if cfg.is_moe: per_layer_mlp = cfg.experts*3*H*cfg.expert_intermediate + H*cfg.experts
    else: per_layer_mlp = 3*H*cfg.intermediate
    return (cfg.vocab_size*H + cfg.layers*(per_layer_attn+per_layer_mlp)) * cfg.dtype_bytes
