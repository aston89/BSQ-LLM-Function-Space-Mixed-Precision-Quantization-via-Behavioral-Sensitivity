import torch
import torch.nn as nn
import torch.nn.functional as F

class BSQLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits=4):
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())

        self.bits = bits

    def quantize(self, w):
        qmax = 2 ** (self.bits - 1) - 1
        qmin = -qmax

        scale = w.abs().max() / (qmax + 1e-8)
        w_q = torch.clamp((w / scale).round(), qmin, qmax) * scale

        return w_q

    def forward(self, x):
        w_q = self.quantize(self.weight)
        return F.linear(x, w_q, self.bias)