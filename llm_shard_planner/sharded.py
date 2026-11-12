import json, os, tempfile
import torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
from .model import rmsnorm, build

def _col(w, rank, ws):  # column-parallel: shard output rows
    return w.chunk(ws, 0)[rank].clone()
def _row(w, rank, ws):  # row-parallel: shard input cols
    return w.chunk(ws, 1)[rank].clone()
def _lin(w):
    m = nn.Linear(w.shape[1], w.shape[0], bias=False); m.weight = nn.Parameter(w); return m

class ShardedLayer(nn.Module):
    """TP for attention/dense MLP; EP (expert slice + all-reduce combine) for MoE.
    ponytail: MoE uses replicated router + all-reduce combine instead of a true
    all-to-all token dispatch; output is identical, wire pattern differs."""
    def __init__(self, cfg, sd, prefix, rank, ws):
        super().__init__(); self.cfg, self.rank, self.ws = cfg, rank, ws
        g = lambda n: sd[f"{prefix}.{n}"]
        self.n1, self.n2 = g("n1"), g("n2")
        self.q, self.k, self.v = _lin(_col(g("q.weight"),rank,ws)), _lin(_col(g("k.weight"),rank,ws)), _lin(_col(g("v.weight"),rank,ws))
        self.o = _lin(_row(g("o.weight"),rank,ws))
        if cfg.is_moe:
            self.router_w = g("router.weight")
            self.my_experts = [e for e in range(cfg.experts) if e % ws == rank]
            self.experts = {}
            for e in self.my_experts:
                self.experts[e] = (_col(g(f"experts.{e}.0.weight"),0,1), _col(g(f"experts.{e}.1.weight"),0,1), _row(g(f"experts.{e}.2.weight"),0,1))
                # experts are not TP-sharded here, only EP-distributed (full copy on owner)
        else:
            self.gate, self.up = _lin(_col(g("gate.weight"),rank,ws)), _lin(_col(g("up.weight"),rank,ws))
            self.down = _lin(_row(g("down.weight"),rank,ws))
    def forward(self, x):
        cfg, ws, rank = self.cfg, self.ws, self.rank; B,T,_ = x.shape; d = cfg.head_dim
        h = rmsnorm(x, self.n1)
        nh, nkv = cfg.heads//ws, cfg.kv_heads//ws
        q = self.q(h).view(B,T,nh,d).transpose(1,2); k = self.k(h).view(B,T,nkv,d).transpose(1,2); v = self.v(h).view(B,T,nkv,d).transpose(1,2)
        if nkv != nh: r = nh//nkv; k = k.repeat_interleave(r,1); v = v.repeat_interleave(r,1)
        a = F.scaled_dot_product_attention(q,k,v,is_causal=T>1)
        y = self.o(a.transpose(1,2).reshape(B,T,-1))
        if ws > 1: dist.all_reduce(y)
        x = x + y
        h = rmsnorm(x, self.n2)
        if cfg.is_moe:
            probs, idx = torch.topk(F.softmax(F.linear(h, self.router_w),-1), cfg.top_k, dim=-1)
            out = torch.zeros_like(h)
            for slot in range(cfg.top_k):
                for e, (gw,uw,dw) in self.experts.items():
                    m = idx[...,slot] == e
                    if m.any(): out[m] += probs[...,slot][m].unsqueeze(-1) * (F.silu(F.linear(h[m],gw))*F.linear(h[m],uw)) @ dw.T
            if ws > 1: dist.all_reduce(out)
            return x + out
        y = self.down(F.silu(self.gate(h))*self.up(h))
        if ws > 1: dist.all_reduce(y)
        return x + y

class ShardedDecoder(nn.Module):
    def __init__(self, cfg, sd, rank, ws):
        super().__init__(); self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden); self.embed.weight = nn.Parameter(sd["embed.weight"].clone())
        self.layers = nn.ModuleList([ShardedLayer(cfg, sd, f"layers.{i}", rank, ws) for i in range(cfg.layers)])
        self.final, self.head_w = sd["final"], sd["head.weight"]
    def forward(self, ids):
        x = self.embed(ids)
        for l in self.layers: x = l(x)
        return F.linear(rmsnorm(x, self.final), self.head_w)

def _worker(rank, ws, cfg_dict, sd_path, ids, out_path, profile_path):
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29517")
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    from .config import ModelConfig
    cfg = ModelConfig(**cfg_dict)
    sd = torch.load(sd_path, weights_only=True)
    ref = build(cfg); ref.load_state_dict(sd)
    sharded = ShardedDecoder(cfg, sd, rank, ws).eval()
    with torch.no_grad():
        expected = ref(ids); got = sharded(ids)
        if profile_path:
            from .profile import profile_decode_step
            prof = profile_decode_step(lambda: sharded(ids[:, :1]))
            if rank == 0: json.dump(prof, open(profile_path,"w"))
    err = (expected-got).abs().max().item()
    if rank == 0: json.dump({"world_size":ws,"max_abs_error":err}, open(out_path,"w"))
    dist.destroy_process_group()

def validate(cfg, world_size=2, seq=8, profile=False):
    """Spawn gloo ranks, compare sharded vs unsharded. Returns result dict."""
    import torch.multiprocessing as mp
    from dataclasses import asdict
    sd = build(cfg).state_dict()
    with tempfile.TemporaryDirectory() as td:
        sd_path, out_path = os.path.join(td,"sd.pt"), os.path.join(td,"out.json")
        prof_path = os.path.join(td,"prof.json") if profile else None
        torch.save(sd, sd_path)
        ids = torch.arange(seq).unsqueeze(0) % cfg.vocab_size
        mp.spawn(_worker, args=(world_size, asdict(cfg), sd_path, ids, out_path, prof_path), nprocs=world_size, join=True)
        res = json.load(open(out_path))
        if prof_path: res["profile"] = json.load(open(prof_path))
        return res
