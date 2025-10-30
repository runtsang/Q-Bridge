from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerNet(nn.Module):
    """
    Simple classical sampler that outputs soft‑maxed input parameters
    and a 4‑dimensional weight vector for a 2‑qubit variational circuit.
    """
    def __init__(self, input_dim: int = 2, weight_dim: int = 4) -> None:
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, input_dim),
        )
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, weight_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        inp = self.input_net(x)
        inp = F.softmax(inp, dim=-1)
        w = self.weight_net(x)
        return {"input_params": inp, "weight_params": w}

def SamplerQNN() -> HybridSamplerNet:
    """
    Factory function compatible with the anchor `SamplerQNN.py`.
    """
    return HybridSamplerNet()

__all__ = ["HybridSamplerNet", "SamplerQNN"]
