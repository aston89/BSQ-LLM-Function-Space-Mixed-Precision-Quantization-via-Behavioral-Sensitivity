import torch
import torch.nn as nn

from bsq.patch_transformers import patch_model_bsq, collect_linear_layers
from bsq.sensitivity import estimate_sensitivity
from bsq.allocation import allocate_bits

class ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)


model = ToyTransformer()

# dummy data
data = [torch.randn(16, 128) for _ in range(20)]

layers = collect_linear_layers(model)

sensitivity = estimate_sensitivity(model, data, layers)

param_counts = {k: v.weight.numel() for k, v in layers.items()}

budget = sum(param_counts.values()) * 4  # target avg 4-bit

bits = allocate_bits(sensitivity, param_counts, budget)

print("bit allocation:", bits)

# patch model (uniform fallback, or extend per-layer bits if you want)
model = patch_model_bsq(model, default_bits=4)

x = torch.randn(4, 128)
y = model(x)

print(y.shape)
