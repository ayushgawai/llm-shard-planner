import time, torch

try:
    import triton, triton.language as tl
    @triton.jit
    def _k(x_ptr, r_ptr, w_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        x = tl.load(x_ptr+row*N+cols, mask=m).to(tl.float32)
        r = tl.load(r_ptr+row*N+cols, mask=m).to(tl.float32)
        s = x + r
        ms = tl.sum(s*s, 0)/N
        w = tl.load(w_ptr+cols, mask=m)
        tl.store(y_ptr+row*N+cols, (s*tl.rsqrt(ms+eps)*w).to(tl.load(y_ptr+row*N+cols, mask=m).dtype), mask=m)
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

def fused_rmsnorm_residual(x, residual, weight, eps=1e-5):
    """RMSNorm(x+residual)*weight. Triton on CUDA, else one fused PyTorch expression."""
    if HAS_TRITON and x.is_cuda:
        y = torch.empty_like(x); n = x.shape[-1]
        _k[(x.numel()//n,)](x, residual, weight, y, n, eps, BLOCK=triton.next_power_of_2(n))
        return y
    s = x + residual
    return s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True)+eps) * weight

def naive(x, residual, weight, eps=1e-5):
    s = x + residual
    return s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True)+eps) * weight

def bench(reps=200, shape=(32, 1024), device="cpu"):
    x = torch.randn(shape, device=device); r = torch.randn(shape, device=device); w = torch.randn(shape[-1], device=device)
    for fn in (naive, fused_rmsnorm_residual):  # warmup
        fn(x, r, w)
    out = {}
    for name, fn in [("naive", naive), ("fused", fused_rmsnorm_residual)]:
        t0 = time.perf_counter()
        for _ in range(reps): fn(x, r, w)
        if device == "cuda": torch.cuda.synchronize()
        out[name+"_ms"] = round((time.perf_counter()-t0)/reps*1e3, 4)
    out["backend"] = "triton" if (HAS_TRITON and device=="cuda") else "pytorch"
    return out
