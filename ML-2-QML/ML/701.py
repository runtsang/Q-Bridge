"""Classical classifier factory with configurable architecture and metadata."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


class QuantumClassifierModel(nn.Module):
    """A flexible feed‑forward classifier mirroring the quantum interface.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    hidden_dim : int | None, optional
        Width of each hidden layer.  Defaults to ``num_features``.
    dropout : float, optional
        Dropout probability applied after every hidden layer.
    residual : bool, optional
        If ``True``, add a residual connection between consecutive hidden layers.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.hidden_dim = hidden_dim or num_features
        self.dropout = dropout
        self.residual = residual

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, self.hidden_dim)
            layers.append(linear)
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, 2)

        # Metadata buffers used by the factory
        self.register_buffer("encoding_indices", torch.arange(num_features))
        self.register_buffer("observables", torch.tensor([0, 1], dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for idx, layer in enumerate(self.feature_extractor):
            if isinstance(layer, nn.Linear):
                residual_tensor = out if self.residual and idx > 0 else None
                out = layer(out)
                if residual_tensor is not None:
                    out = out + residual_tensor
            else:
                out = layer(out)
        return self.classifier(out)

    def get_weight_sizes(self) -> List[int]:
        """Return a list of parameter counts for each learnable tensor."""
        return [param.numel() for param in self.parameters()]

    def get_encoding_indices(self) -> List[int]:
        """Return the indices of the input features that are encoded."""
        return list(self.encoding_indices)

    def get_observables(self) -> List[int]:
        """Return a placeholder list of observable identifiers."""
        return list(self.observables)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int | None = None,
    dropout: float = 0.0,
    residual: bool = False,
) -> Tuple[QuantumClassifierModel, Iterable[int], Iterable[int], List[int]]:
    """Construct a classical classifier and return metadata.

    This mirrors the original quantum helper interface but adds optional
    dropout and residual connections.

    Returns
    -------
    model : QuantumClassifierModel
        The constructed PyTorch model.
    encoding : Iterable[int]
        Indices of the input features that are encoded.
    weight_sizes : Iterable[int]
        Number of learnable parameters in each weight tensor.
    observables : List[int]
        Placeholder observable identifiers for compatibility.
    """
    model = QuantumClassifierModel(
        num_features, depth, hidden_dim, dropout, residual
    )
    return (
        model,
        model.get_encoding_indices(),
        model.get_weight_sizes(),
        model.get_observables(),
    )


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
