import torch
from fusionguard import FusionGuard

if __name__ == "__main__":
    guard = FusionGuard(dim=512, hidden_dim=2048)
    x = torch.randn(32, 512)
    y = guard(x)
    print("Output shape:", y.shape)
