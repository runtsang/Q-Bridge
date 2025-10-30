"""Classical sampler network extended with deeper architecture and KL divergence metric."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen064(nn.Module):
    """A deeper neural sampler that outputs a probability distribution over 2 classes."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.net(inputs)
        return F.softmax(logits, dim=-1)

    def kl_divergence(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between output distribution and target."""
        eps = 1e-12
        probs = torch.clamp(probs, eps, 1.0)
        target = torch.clamp(target, eps, 1.0)
        return torch.sum(target * torch.log(target / probs), dim=-1).mean()

def SamplerQNNGen064() -> SamplerQNNGen064:
    """Factory returning a fresh instance of SamplerQNNGen064."""
    return SamplerQNNGen064()

__all__ = ["SamplerQNNGen064"]
