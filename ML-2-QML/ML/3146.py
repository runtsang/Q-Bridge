"""
HybridEstimatorQNN – classical implementation.

Implements a lightweight regression network that emulates the
convolution/pooling pattern of the QCNN reference.  Each “convolution”
is realized as a fully connected layer with a tanh non‑linearity,
followed by a pooling step that reduces dimensionality.  The final
sigmoid head maps the extracted features to a single scalar output.
"""

from __future__ import annotations

import torch
from torch import nn

class HybridEstimatorQNN(nn.Module):
    """
    Classical regression network mirroring the QCNN structure.

    Parameters
    ----------
    input_dim : int, default 2
        Dimensionality of the input feature vector.
    hidden_dims : Sequence[int], default [16, 16, 12, 8, 4, 4]
        Layer widths that follow the convolution → pooling → convolution
        pattern of the QCNN ansatz.

    Notes
    -----
    The network is fully differentiable and can be trained with any
    PyTorch optimiser.  It is designed to be drop‑in compatible with
    the original EstimatorQNN interface, facilitating quick
    experimentation without a quantum backend.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] = [16, 16, 12, 8, 4, 4]
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh()
        )
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.Tanh()
                )
            )
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.head(x))

__all__ = ["HybridEstimatorQNN"]
