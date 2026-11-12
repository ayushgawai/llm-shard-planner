from torch.profiler import profile, ProfilerActivity

def profile_decode_step(fn):
    """One step under torch.profiler, split into compute vs communication by op name."""
    fn()  # warmup
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    compute = comm = 0.0
    for e in prof.key_averages():
        t = e.self_cpu_time_total
        if any(s in e.key.lower() for s in ("c10d", "all_reduce", "allreduce", "all_to_all", "all_gather", "gloo")):
            comm += t
        else:
            compute += t
    total = compute + comm or 1.0
    return {"compute_us": round(compute,1), "comm_us": round(comm,1),
            "comm_pct": round(100*comm/total,2),
            "top_ops": [{"op": e.key, "self_cpu_us": round(e.self_cpu_time_total,1)} for e in sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)[:8]]}
