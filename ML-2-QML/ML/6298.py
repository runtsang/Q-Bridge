"""Enhanced classical sampler network with Bayesian dropout and uncertainty output."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class SamplerModule(nn.Module):
    """
    Drop‑in replacement for the original SamplerQNN.
    Architecture:
        * Feature extractor: Linear → ReLU → Dropout
        * Probability head: Linear → softmax
        * Uncertainty head: Linear → sigmoid
    The Bayesian dropout is realized via a Bernoulli mask applied during training,
    and the uncertainty head outputs a learnable probability of mis‑classification.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 8,
        out_features: int = 2,
        dropout_prob: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )
        self.prob_head = nn.Linear(hidden_features, out_features)
        self.unc_head = nn.Linear(hidden_features, 1)
        self.dropout_prob = dropout_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns a tuple (probs, uncertainty).
        """
        feats = self.feature_extractor(x)
        probs = F.softmax(self.prob_head(feats), dim=-1)
        uncertainty = torch.sigmoid(self.unc_head(feats))
        return probs, uncertainty

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample from the categorical distribution defined by *probs*.
        """
        probs, _ = self.forward(x)
        m = torch.distributions.Categorical(probs)
        return m.sample()

    def training_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Single training step returning the loss.
        """
        probs, uncertainty = self.forward(x)
        loss = criterion(probs, y)
        loss += 0.1 * uncertainty.mean()
        return loss

def SamplerQNN() -> SamplerModule:
    """Factory returning a ready‑to‑train SamplerModule."""
    return SamplerModule()
