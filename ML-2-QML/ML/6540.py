import torch
import torch.nn as nn
import torch.nn.functional as F

def SamplerQNN(in_features: int = 2, hidden_sizes: list[int] | None = None, out_features: int = 2):
    """
    Configurable classical sampler network.
    """
    if hidden_sizes is None:
        hidden_sizes = [4]
    layers = []
    prev = in_features
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_features))
    net = nn.Sequential(*layers)

    class SamplerModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.softmax(self.net(x), dim=-1)

    return SamplerModule()
