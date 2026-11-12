import torch
from llm_shard_planner.config import ModelConfig, DeviceProfile
from llm_shard_planner.cost import layer_cost, collective_time
from llm_shard_planner.planner import enumerate_plans
from llm_shard_planner.fused import fused_rmsnorm_residual, naive

DEV = DeviceProfile()

def test_cost_positive_and_kv_grows():
    c1 = layer_cost(ModelConfig(), DEV, seq=128, context=128)
    c2 = layer_cost(ModelConfig(), DEV, seq=128, context=1024)
    assert c1.flops > 0 and c1.hbm_bytes > 0 and c2.kv_bytes > c1.kv_bytes

def test_collective_alpha_beta():
    assert collective_time(1e6, "all_reduce", 1, DEV) == 0
    assert collective_time(1e6, "all_reduce", 4, DEV) > collective_time(1e3, "all_reduce", 4, DEV) > 0
    assert collective_time(1e6, "all_to_all", 4, DEV) > 0

def test_moe_uses_topk_not_all_experts():
    moe = ModelConfig(experts=8, top_k=2, expert_intermediate=128)
    one = ModelConfig(experts=8, top_k=1, expert_intermediate=128)
    assert layer_cost(moe, DEV, seq=16).flops > layer_cost(one, DEV, seq=16).flops

def test_memory_rejection():
    tiny_dev = DeviceProfile(hbm_gb=0.01)
    big = ModelConfig(hidden=1024, layers=32, heads=16, kv_heads=16, intermediate=4096, vocab_size=32000)
    feas, infeas = enumerate_plans(big, tiny_dev, devices=8, seq=512)
    assert feas == [] and infeas

def test_planner_ranks_and_finds_feasible():
    feas, _ = enumerate_plans(ModelConfig(), DEV, devices=8, seq=256)
    assert feas and feas == sorted(feas, key=lambda p: p.prefill_latency_s)
    assert all(p.mem_per_device_gb <= DEV.hbm_gb for p in feas)

def test_fused_rmsnorm():
    x, r, w = torch.randn(8, 64), torch.randn(8, 64), torch.randn(64)
    assert torch.allclose(fused_rmsnorm_residual(x, r, w), naive(x, r, w), atol=1e-5)

def test_sharded_matches_unsharded_dense():
    from llm_shard_planner.sharded import validate
    assert validate(ModelConfig(), world_size=2)["max_abs_error"] < 1e-4

def test_sharded_matches_unsharded_moe():
    from llm_shard_planner.sharded import validate
    cfg = ModelConfig(experts=4, top_k=2)
    assert validate(cfg, world_size=2)["max_abs_error"] < 1e-4
