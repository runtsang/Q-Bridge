"""Hybrid classical sampler‑classifier model combining the seed SamplerQNN and QuantumClassifierModel architectures.

The class exposes a sampler network followed by a classifier network, mirroring the
quantum‑classical interface of the original seeds while adding a joint forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables: List[int] = list(range(2))
    return network, encoding, weight_sizes, observables


class SamplerQNNGen169(nn.Module):
    """Hybrid sampler‑classifier network.

    The sampler produces a probability distribution over the input space,
    which is then fed as a feature vector into a classifier.  This mirrors
    the quantum sampler + classifier workflow while remaining fully classical.
    """

    def __init__(self, num_features: int = 2, depth: int = 2) -> None:
        super().__init__()
        self.sampler_net = nn.Sequential(
            nn.Linear(num_features, 4),
            nn.Tanh(),
            nn.Linear(4, num_features),
        )
        self.classifier_net, _, _, _ = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return classifier logits."""
        sample_probs = F.softmax(self.sampler_net(x), dim=-1)
        return self.classifier_net(sample_probs)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Return a discrete sample drawn from the sampler distribution."""
        probs = F.softmax(self.sampler_net(x), dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return posterior class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


__all__ = ["SamplerQNNGen169"]
