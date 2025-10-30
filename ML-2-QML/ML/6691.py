import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BayesianLinear(nn.Module):
    """Probabilistic linear layer that learns a mean and log‑variance for each weight."""
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.randn(out_features))
        self.bias_logvar = nn.Parameter(torch.randn(out_features))
        self.prior_sigma = prior_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps_w = torch.randn_like(self.weight_mu)
        eps_b = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + torch.exp(self.weight_logvar) * eps_w
        bias = self.bias_mu + torch.exp(self.bias_logvar) * eps_b
        return F.linear(x, weight, bias)

class HybridNet(nn.Module):
    """Shared CNN backbone with a Bayesian linear head for a classical baseline."""
    def __init__(self):
        super().__init__()
        # Convolutional backbone (identical to the quantum version)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Determine feature size after conv layers
        dummy = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            out = self._extract_features(dummy)
        in_features = out.shape[1]
        # Bayesian head
        self.bayes_head = BayesianLinear(in_features, 1)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(x)
        logits = self.bayes_head(features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridNet"]
