import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerModule(nn.Module):
    """
    A deeper classical sampler with batch‑norm, ReLU, and dropout layers.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

def SamplerQNN():
    """
    Return an instance of the upgraded classical sampler.
    """
    return SamplerModule()

__all__ = ["SamplerQNN"]
