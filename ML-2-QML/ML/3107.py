import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNHybrid(nn.Module):
    """Classical sampler network with temperature scaling."""
    def __init__(self, in_features=2, hidden=4, out_features=2, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.net(inputs) / self.temperature
        return F.softmax(logits, dim=-1)

def SamplerQNN() -> SamplerQNNHybrid:
    return SamplerQNNHybrid()
