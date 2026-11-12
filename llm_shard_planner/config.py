import json
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    name: str = "tiny-dense"
    vocab_size: int = 512
    hidden: int = 128
    layers: int = 4
    heads: int = 4
    kv_heads: int = 2
    intermediate: int = 256
    experts: int = 0          # 0 = dense MLP
    expert_intermediate: int = 128
    top_k: int = 0
    dtype_bytes: int = 2
    @property
    def head_dim(self): return self.hidden // self.heads
    @property
    def is_moe(self): return self.experts > 0

@dataclass
class DeviceProfile:
    name: str = "generic-80gb"
    hbm_gb: float = 80.0
    hbm_bw_gbs: float = 2000.0
    peak_tflops: float = 300.0
    link_bw_gbs: float = 300.0
    link_latency_us: float = 5.0

def _load(cls, path):
    with open(path) as f: data = json.load(f)
    known = cls.__dataclass_fields__
    return cls(**{k: v for k, v in data.items() if k in known})

def load_model(path): return _load(ModelConfig, path)
def load_device(path): return _load(DeviceProfile, path)
