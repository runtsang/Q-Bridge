"""Hybrid classical regressor with an optional quantum feature extractor.

The model is fully classical but can wrap a quantum layer that
produces additional features.  The architecture is fully
configurable: feature extractor depth, hidden sizes, and the
quantum layer can be swapped or omitted.
"""

import torch
from torch import nn
from typing import Tuple, Optional

class EstimatorQNN(nn.Module):
    """
    A classical feed‑forward regressor that optionally appends quantum
    features produced by a supplied ``nn.Module``.

    Parameters
    ----------
    input_dim : int, default 2
        Dimension of the input features.
    hidden_dims : Tuple[int,...], default (8, 4)
        Sizes of the hidden layers in the feature extractor.
    output_dim : int, default 1
        Dimension of the regression output.
    quantum_layer : Optional[nn.Module], default None
        A PyTorch module that accepts the output of the feature
        extractor and returns a tensor of quantum features.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Tuple[int,...] = (8, 4),
        output_dim: int = 1,
        quantum_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.feature_extractor = nn.Sequential(*layers)
        self.quantum_layer = quantum_layer
        # If a quantum layer is used, its output will be concatenated
        # to the feature extractor output.  The head must therefore
        # accept the enlarged dimensionality.
        head_input = prev + (quantum_layer.out_features if quantum_layer else 0)
        self.head = nn.Linear(head_input, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor, optional quantum
        layer, and final linear head."""
        feat = self.feature_extractor(x)
        if self.quantum_layer is not None:
            qfeat = self.quantum_layer(feat)
            feat = torch.cat([feat, qfeat], dim=-1)
        return self.head(feat)

    @staticmethod
    def default_quantum_layer() -> nn.Module:
        """Return a lightweight dummy quantum layer that generates
        non‑linear features.  This can be used for quick prototyping
        when a true quantum backend is not available."""
        class DummyQuantum(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Simple feature map: sin and cos of each input dimension
                return torch.stack([torch.sin(x), torch.cos(x)], dim=-1)

            @property
            def out_features(self) -> int:
                # Two features per input dimension
                return 2 * x.shape[-1]

        return DummyQuantum()
