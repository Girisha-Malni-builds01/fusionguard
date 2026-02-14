import torch
from .block import MLPBlock
from .fused import FusedMLPBlock
from .quant import dynamic_int8
from .benchmark import measure

class FusionGuard:

    def __init__(self, dim=512, hidden_dim=2048, device=None, runs=50):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.runs = runs
        self.best = None
        self._initialize()

    def _initialize(self):
        x = torch.randn(32, self.dim)

        variants = {
            "fp32_unfused": MLPBlock(self.dim, self.hidden_dim),
            "fp32_fused": FusedMLPBlock(self.dim, self.hidden_dim),
            "int8_unfused": dynamic_int8(MLPBlock(self.dim, self.hidden_dim)),
            "int8_fused": dynamic_int8(FusedMLPBlock(self.dim, self.hidden_dim)),
        }

        timings = {}

        for name, model in variants.items():
            try:
                t = measure(model, self.device, x, self.runs)
                timings[name] = t
            except Exception:
                continue

        self.best_name = min(timings, key=timings.get)
        self.best = variants[self.best_name].to(self.device)

        print("FusionGuard Selection Results (ms):")
        for k, v in timings.items():
            print(f"{k}: {v*1000:.3f}")
        print("Selected:", self.best_name)

    def __call__(self, x):
        return self.best(x.to(self.device))
