"""Classical quantum-inspired classifier with advanced neural architecture."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifierModel(nn.Module):
    """
    A feed‑forward neural network mimicking the interface of the quantum classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    depth : int
        Number of hidden layers.
    hidden_dim : int, optional
        Width of each hidden layer. Defaults to ``num_features``.
    dropout : float, optional
        Dropout probability applied after each hidden layer.
    use_batchnorm : bool, optional
        Whether to insert a batch‑norm layer after each linear block.
    residual : bool, optional
        Whether to add residual connections between consecutive blocks.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        residual: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or num_features
        layers: List[nn.Module] = []

        in_dim = num_features
        for i in range(depth):
            linear = nn.Linear(in_dim, hidden_dim, bias=True)
            block = [linear, nn.ReLU()]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0.0:
                block.append(nn.Dropout(dropout))
            layers.append(nn.Sequential(*block))
            in_dim = hidden_dim

        self.body = nn.Sequential(*layers)
        self.residual = residual and depth > 1
        self.head = nn.Linear(in_dim, 2, bias=True)

        # Metadata used by the legacy interface
        self._encoding = list(range(num_features))
        self._weight_sizes = self._compute_weight_sizes()
        self._observables = list(range(2))

    def _compute_weight_sizes(self) -> List[int]:
        sizes = []
        for m in self.body:
            for sub in m:
                if isinstance(sub, nn.Linear):
                    sizes.append(sub.weight.numel() + sub.bias.numel())
        sizes.append(self.head.weight.numel() + self.head.bias.numel())
        return sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for i, block in enumerate(self.body):
            residual = out
            out = block(out)
            if self.residual:
                out = out + residual
        logits = self.head(out)
        return logits

    @property
    def encoding(self) -> List[int]:
        """Indices of features that are explicitly used (all of them)."""
        return self._encoding

    @property
    def weight_sizes(self) -> List[int]:
        """Number of trainable parameters per linear layer."""
        return self._weight_sizes

    @property
    def observables(self) -> List[int]:
        """Placeholder for compatibility with the quantum API."""
        return self._observables

    @classmethod
    def build_classifier_circuit(
        cls, num_features: int, depth: int
    ) -> Tuple["QuantumClassifierModel", List[int], List[int], List[int]]:
        """
        Factory that preserves the original signature while returning an instance
        and the metadata required by downstream code.
        """
        model = cls(num_features, depth)
        return model, model.encoding, model.weight_sizes, model.observables


__all__ = ["QuantumClassifierModel"]
