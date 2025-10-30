"""
Hybrid classical classifier mirroring the quantum helper interface.
Introduces clipping, scaling, and a flexible depth parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


@dataclass
class ClassifierLayerParameters:
    """
    Parameters for a single feed‑forward layer.
    * weight_shape: (out_features, in_features)
    * bias_shape: (out_features,)
    * scale: multiplicative factor applied after activation
    * shift: additive offset applied after scaling
    """
    weight: torch.Tensor
    bias: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor


def _clip_tensor(tensor: torch.Tensor, bound: float) -> torch.Tensor:
    """Clamp tensor values to the interval [-bound, bound]."""
    return tensor.clamp(-bound, bound)


def _layer_from_params(params: ClassifierLayerParameters, *, clip: bool = True) -> nn.Module:
    """Create a single linear + activation + scaling layer."""
    linear = nn.Linear(params.weight.size(1), params.weight.size(0))
    with torch.no_grad():
        linear.weight.copy_(params.weight)
        linear.bias.copy_(params.bias)

    activation = nn.Tanh()

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", params.scale)
            self.register_buffer("shift", params.shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class HybridClassifier(nn.Module):
    """
    A flexible feed‑forward network that mimics the quantum interface.
    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dim : int
        Width of hidden layers.
    depth : int
        Number of hidden layers.
    clip : bool, default=True
        Whether to clip weights and biases to a reasonable range.
    """
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, clip: bool = True) -> None:
        super().__init__()
        self.depth = depth
        self.clip = clip

        layers: List[nn.Module] = []

        # Input layer
        in_dim = input_dim
        for l in range(depth):
            out_dim = hidden_dim if l < depth - 1 else 1  # final layer outputs a single logit
            weight = torch.randn(out_dim, in_dim)
            bias = torch.randn(out_dim)
            scale = torch.ones(out_dim)
            shift = torch.zeros(out_dim)

            if clip:
                weight = _clip_tensor(weight, 5.0)
                bias = _clip_tensor(bias, 5.0)

            params = ClassifierLayerParameters(weight, bias, scale, shift)
            layer = _layer_from_params(params, clip=clip)
            layers.append(layer)
            in_dim = out_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def metadata(self) -> Tuple[Iterable[int], Iterable[int], List[int]]:
        """
        Return metadata compatible with the quantum helper interface:
        - encoding: list of input feature indices (identity mapping)
        - weight_sizes: cumulative number of trainable parameters per layer
        - observables: list of output indices (here just [0] for binary logit)
        """
        encoding = list(range(self.network[0].linear.in_features))
        weight_sizes = []
        cumulative = 0
        for module in self.network:
            if isinstance(module, nn.Linear):
                n_params = module.weight.numel() + module.bias.numel()
                cumulative += n_params
                weight_sizes.append(cumulative)
        observables = [0]
        return encoding, weight_sizes, observables


def build_classifier_circuit(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    clip: bool = True,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a HybridClassifier and expose the same metadata as the quantum version.
    """
    model = HybridClassifier(input_dim, hidden_dim, depth, clip=clip)
    encoding, weight_sizes, observables = model.metadata()
    return model, encoding, weight_sizes, observables


__all__ = [
    "ClassifierLayerParameters",
    "HybridClassifier",
    "build_classifier_circuit",
]
