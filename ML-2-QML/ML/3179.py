import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN__gen111(nn.Module):
    """
    Classical sampler network with a regression head.
    Input shape: (batch, 2) – two feature values.
    The network first produces a probability distribution over two outcomes
    using a small feed‑forward network, then maps that distribution to a
    scalar prediction via a linear head.
    """

    def __init__(self) -> None:
        super().__init__()
        # Sampler core
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )
        # Regression head applied to the sampler output
        self.head = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass: compute softmax probabilities, then regress.
        """
        probs = F.softmax(self.sampler(inputs), dim=-1)
        return self.head(probs).squeeze(-1)

__all__ = ["SamplerQNN__gen111"]
