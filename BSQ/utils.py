import torch

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())