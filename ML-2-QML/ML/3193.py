"""Hybrid classical model combining a fully‑connected layer and a sampler network.

The class implements a two‑stage feed‑forward network where the first stage
acts as a fully‑connected layer (linear + tanh) and the second stage is a
softmax sampler.  The weights are learnable PyTorch parameters and the
forward pass can be used as a drop‑in replacement for the original FCL
and SamplerQNN modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFCLSampler(nn.Module):
    def __init__(self):
        super().__init__()
        # Fully connected stage
        self.fcl = nn.Sequential(
            nn.Linear(1, 1),
            nn.Tanh()
        )
        # Sampler stage
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a probability distribution."""
        # Expect thetas shape (batch, 1)
        x = self.fcl(thetas)
        # Concatenate with an auxiliary input (e.g., zeros) to feed sampler
        aux = torch.zeros_like(x)
        inp = torch.cat([x, aux], dim=-1)  # shape (batch, 2)
        probs = F.softmax(self.sampler(inp), dim=-1)
        return probs

    def run(self, thetas):
        """Convenience wrapper mimicking the original FCL run signature."""
        with torch.no_grad():
            thetas_tensor = torch.tensor(thetas, dtype=torch.float32).view(-1, 1)
            probs = self.forward(thetas_tensor)
            expectation = probs.sum(dim=1)  # example aggregation
            return expectation.numpy()

__all__ = ["HybridFCLSampler"]
