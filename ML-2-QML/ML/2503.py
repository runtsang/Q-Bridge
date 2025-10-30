"""Hybrid classical classifier that optionally uses a quantum kernel as a feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Tuple, Sequence, Callable, List

class HybridClassifier(nn.Module):
    """A feed‑forward classifier that can prepend a quantum or classical kernel feature map.

    Parameters
    ----------
    num_features : int
        Dimensionality of the raw input.
    hidden_dims : Sequence[int]
        Sizes of hidden layers.
    use_quantum_kernel : bool
        When True the input is first mapped by a callable ``kernel`` which must return a
        1‑D feature vector. The kernel is expected to be a quantum kernel implemented
        in the QML module.
    kernel : Callable[[torch.Tensor], torch.Tensor], optional
        Callable that transforms a batch of inputs into feature space.
    gamma : float, optional
        RBF kernel bandwidth used when ``use_quantum_kernel`` is False.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dims: Sequence[int] = (64, 32),
        use_quantum_kernel: bool = False,
        kernel: Callable[[torch.Tensor], torch.Tensor] | None = None,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_quantum_kernel = use_quantum_kernel
        self.kernel = kernel
        self.gamma = gamma

        # Build the classifier backbone
        layers: List[nn.Module] = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        self.backbone = nn.Sequential(*layers)

        # Store weight sizes for introspection
        self.weight_sizes = [p.numel() for p in self.backbone.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        If ``use_quantum_kernel`` is True, ``x`` is first passed through the kernel
        function before being fed to the backbone.
        """
        if self.use_quantum_kernel and self.kernel is not None:
            x = self.kernel(x)  # shape (batch, feat)
        else:
            # Default to raw input; optional RBF feature map can be added here.
            if x.ndim == 1:
                x = x.unsqueeze(0)
        return self.backbone(x)

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Return a classical network and metadata mirroring the quantum helper."""
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

__all__ = ["HybridClassifier"]
