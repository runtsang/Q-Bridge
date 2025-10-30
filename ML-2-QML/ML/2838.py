import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable

class SamplerQNN(nn.Module):
    """Classical approximation of the quantum sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridFCL(nn.Module):
    """Hybrid fully connected layer with optional classical sampler."""
    def __init__(self, n_features: int = 1, use_sampler: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = SamplerQNN()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return expectation of the tanh activation and, if enabled,
        a weighted combination with the classical sampler output."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        if self.use_sampler:
            inp = torch.as_tensor(thetas[:2], dtype=torch.float32).view(1, -1)
            probs = self.sampler(inp).detach().numpy().flatten()
            combined = expectation.detach().numpy() + probs.sum() * 0.1
            return np.array([combined])
        return expectation.detach().numpy()

def FCL() -> HybridFCL:
    """Factory function returning a HybridFCL instance."""
    return HybridFCL()

__all__ = ["HybridFCL", "FCL"]
