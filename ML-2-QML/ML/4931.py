import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerQNNCombined(nn.Module):
    """Purely classical sampler network.
    Mirrors the structure of the original SamplerQNN but kept lightweight.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


__all__ = ["SamplerQNNCombined"]
