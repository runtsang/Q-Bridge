import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerEstimatorQNN(nn.Module):
    """
    A hybrid classical neural network that first samples from a low‑dimensional
    distribution using a small feed‑forward sampler, then regresses a target
    value using an additional feed‑forward estimator.  The sampler and estimator
    share the input space and can be trained jointly or separately.
    """
    def __init__(self, sampler_hidden: int = 4, estimator_hidden: int = 8) -> None:
        super().__init__()
        # Sampler: maps 2‑dimensional input to a 2‑dimensional probability vector
        self.sampler = nn.Sequential(
            nn.Linear(2, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2)
        )
        # Estimator: maps 2‑dimensional sampler output to a single regression target
        self.estimator = nn.Sequential(
            nn.Linear(2, estimator_hidden),
            nn.Tanh(),
            nn.Linear(estimator_hidden, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: produce a probability distribution from the sampler and then
        predict a scalar target using the estimator.
        """
        probs = F.softmax(self.sampler(x), dim=-1)
        # Optionally one could sample from probs, but we use the softmax vector directly
        out = self.estimator(probs)
        return out
