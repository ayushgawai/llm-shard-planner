import torch, torch.nn as nn, torch.nn.functional as F

def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        H = cfg.hidden
        self.n1 = nn.Parameter(torch.ones(H)); self.n2 = nn.Parameter(torch.ones(H))
        self.q = nn.Linear(H, cfg.heads*cfg.head_dim, bias=False)
        self.k = nn.Linear(H, cfg.kv_heads*cfg.head_dim, bias=False)
        self.v = nn.Linear(H, cfg.kv_heads*cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.heads*cfg.head_dim, H, bias=False)
        if cfg.is_moe:
            self.router = nn.Linear(H, cfg.experts, bias=False)
            self.experts = nn.ModuleList([nn.ModuleList([
                nn.Linear(H, cfg.expert_intermediate, bias=False),
                nn.Linear(H, cfg.expert_intermediate, bias=False),
                nn.Linear(cfg.expert_intermediate, H, bias=False)]) for _ in range(cfg.experts)])
        else:
            self.gate = nn.Linear(H, cfg.intermediate, bias=False)
            self.up = nn.Linear(H, cfg.intermediate, bias=False)
            self.down = nn.Linear(cfg.intermediate, H, bias=False)
    def mlp(self, x):
        cfg = self.cfg
        if not cfg.is_moe: return self.down(F.silu(self.gate(x)) * self.up(x))
        probs, idx = torch.topk(F.softmax(self.router(x), -1), cfg.top_k, dim=-1)
        out = torch.zeros_like(x)
        for slot in range(cfg.top_k):
            for e in range(cfg.experts):
                m = idx[..., slot] == e
                if not m.any(): continue
                g, u, dn = self.experts[e]
                out[m] += probs[..., slot][m].unsqueeze(-1) * dn(F.silu(g(x[m])) * u(x[m]))
        return out
    def forward(self, x):
        cfg = self.cfg; B, T, _ = x.shape; d = cfg.head_dim
        h = rmsnorm(x, self.n1)
        q = self.q(h).view(B,T,cfg.heads,d).transpose(1,2)
        k = self.k(h).view(B,T,cfg.kv_heads,d).transpose(1,2)
        v = self.v(h).view(B,T,cfg.kv_heads,d).transpose(1,2)
        if cfg.kv_heads != cfg.heads:  # GQA repeat
            r = cfg.heads//cfg.kv_heads; k = k.repeat_interleave(r,1); v = v.repeat_interleave(r,1)
        a = F.scaled_dot_product_attention(q,k,v,is_causal=T>1)
        x = x + self.o(a.transpose(1,2).reshape(B,T,-1))
        return x + self.mlp(rmsnorm(x, self.n2))

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.layers)])
        self.final = nn.Parameter(torch.ones(cfg.hidden))
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
    def forward(self, ids):
        x = self.embed(ids)
        for l in self.layers: x = l(x)
        return self.head(rmsnorm(x, self.final))

def build(cfg, seed=0):
    torch.manual_seed(seed); m = Decoder(cfg)
    with torch.no_grad():
        for p in m.parameters(): p.mul_(0.5)  # keep activations tame
    return m.eval()
