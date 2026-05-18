import torch.nn as nn
from .quant_linear import BSQLinear

def patch_model_bsq(model, default_bits=4):
    """
    Recursively replaces nn.Linear with BSQLinear.
    """

    for name, module in model.named_children():

        if isinstance(module, nn.Linear):
            setattr(model, name, BSQLinear(module, bits=default_bits))

        else:
            patch_model_bsq(module, default_bits)

    return model


def collect_linear_layers(model):
    layers = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            layers[name] = m
    return layers