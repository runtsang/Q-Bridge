"""
Establishes a flexible, drop‑out enabled feed‑forward regressor with optional
batch‑normalisation.  The architecture can be tuned via keyword arguments,
allowing the model to adapt to higher‑dimensional inputs or deeper feature
extractors.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence, Tuple, Iterable


class EstimatorQNN(nn.Module):
    """
    A configurable feed‑forward regression network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dims : Sequence[int]
        Sizes of the hidden layers.  The default reproduces the original
        2→8→4→1 topology.
    output_dim : int
        Size of the output.  Default is 1 for a scalar regression task.
    dropout : float, optional
        Drop‑out probability applied after each hidden layer.
    use_batchnorm : bool, default False
        Whether to insert a BatchNorm1d layer before the activation.

    Notes
    -----
    * The model outputs a raw tensor; callers can apply a loss function
      (e.g. MSELoss) as needed.
    * ``predict`` returns a detached, CPU‑resident NumPy array for
      convenient downstream usage.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (8, 4),
        output_dim: int = 1,
        dropout: float = 0.1,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        layers: Iterable[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

    def predict(self, inputs: torch.Tensor | Iterable[float]) -> torch.Tensor:
        """
        Forward pass returning a NumPy array.

        Parameters
        ----------
        inputs : torch.Tensor or iterable of float
            Input data.  If an iterable is provided, it is converted to
            a 2‑D tensor with shape ``(batch, input_dim)``.
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            out = self.forward(inputs)
        return out.cpu().numpy()

__all__ = ["EstimatorQNN"]
