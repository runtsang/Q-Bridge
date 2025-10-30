import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a feed‑forward classifier that mirrors the quantum interface.
    The network consists of `depth` hidden layers each returning `num_features` units,
    followed by a binary head.  Metadata (encoding, weight sizes, observables) is
    returned to keep parity with the quantum implementation.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class SamplerModule(nn.Module):
    """
    A lightweight sampler that maps logits to a probability simplex,
    mirroring the quantum SamplerQNN but implemented classically.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class QuantumClassifierModel(nn.Module):
    """
    Classical classifier that exposes the same public API as its quantum counterpart.
    It builds a feed‑forward network, optionally followed by a sampler module.
    """
    def __init__(self, num_features: int, depth: int, use_sampler: bool = True) -> None:
        super().__init__()
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(num_features, depth)
        self.use_sampler = use_sampler
        self.sampler = SamplerModule() if use_sampler else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        if self.use_sampler:
            return self.sampler(logits)
        return logits

    def get_encoding(self) -> Iterable[int]:
        return self.encoding

    def get_params(self) -> Iterable[int]:
        return self.weight_sizes

__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "SamplerModule"]
