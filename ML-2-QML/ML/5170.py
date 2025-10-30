from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head used by both classical and quantum models."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """
    Construct a classical feed‑forward classifier that mirrors the structure
    of the quantum ansatz used in the QML partner.  The returned tuple
    contains the network, an encoding list (indices of input features),
    a list of weight counts for each layer, and a list of observable
    indices (here simply `[0, 1]` for the two output classes).
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
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


class HybridClassifier(nn.Module):
    """
    Classic hybrid classifier that can be used as a drop‑in replacement
    for the quantum version.  It accepts a feature tensor of shape
    `(batch, num_features)` and outputs a probability distribution over
    two classes.  The `shift` parameter allows fine‑tuning of the sigmoid
    threshold.
    """
    def __init__(self, num_features: int, depth: int = 2, shift: float = 0.0) -> None:
        super().__init__()
        self.backbone, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )
        # Head produces a single logit; sigmoid + shift produces probability
        self.head = nn.Linear(num_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        probs = HybridFunction.apply(logits, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridFunction", "HybridClassifier", "build_classifier_circuit"]
