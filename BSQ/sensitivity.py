import torch
import torch.nn.functional as F

def _jvp_proxy(model, x, eps=1e-3):
    """
    Lightweight functional sensitivity proxy:
    approximates ||Jv||^2 via finite perturbation.
    """
    x = x.detach()

    v = torch.randint_like(x, low=0, high=2).float() * 2 - 1
    x1 = x + eps * v
    x2 = x - eps * v

    y1 = model(x1)
    y2 = model(x2)

    return (y1 - y2).pow(2).mean()


@torch.no_grad()
def estimate_sensitivity(model, dataloader, layers, device="cuda", samples=64):
    """
    Returns per-layer behavioral sensitivity scores.

    layers: dict name -> module
    """

    model.eval()
    model = model.to(device)

    scores = {name: 0.0 for name in layers.keys()}

    for i, x in enumerate(dataloader):
        if i >= samples:
            break

        x = x.to(device)

        base_out = model(x)

        for name, layer in layers.items():
            # hook-style perturbation proxy
            layer_out = layer(x)
            scores[name] += (base_out - layer_out).pow(2).mean().item()

    for k in scores:
        scores[k] /= samples

    return scores