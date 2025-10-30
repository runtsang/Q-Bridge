"""Hybrid LSTM implementation with classical and quantum interfaces.

This module defines a `HybridQLSTM` class that can operate as a standard
PyTorch LSTM or as a quantum‑enhanced LSTM depending on the `n_qubits`
argument.  The classical variant supports dropout and layer‑normalisation,
while the quantum variant builds on the torchquantum circuit described
in the QML seed.  An accompanying `FastEstimator` helper is provided
to evaluate a batch of inputs and optionally add Gaussian shot noise.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridQLSTM(nn.Module):
    """Hybrid LSTM that can be either classical or quantum.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dim : int
        Dimensionality of hidden state.
    n_qubits : int, default=0
        If ``>0`` a quantum LSTM is used; otherwise a standard
        ``nn.LSTM``.  The quantum implementation is provided in the
        parallel ``qml`` module.
    dropout : float, default=0.0
        Dropout probability applied to the output of each cell.
    layer_norm : bool, default=False
        Whether to apply layer normalisation after each gate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = dropout
        self.layer_norm = layer_norm

        if n_qubits > 0:
            raise ValueError(
                "Quantum mode is not available in the classical module; "
                "use the corresponding QML module."
            )

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=False)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the hybrid LSTM.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, input_dim)``.
        states : tuple, optional
            Tuple of ``(h_0, c_0)`` hidden states.

        Returns
        -------
        outputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, hidden_dim)``.
        (h_n, c_n) : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs, (hn, cn) = self.lstm(inputs, (hx, cx))
        outputs = self.dropout_layer(outputs)
        outputs = self.ln(outputs)
        return outputs, (hn.squeeze(0), cn.squeeze(0))

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )

class FastEstimator:
    """Lightweight estimator that can evaluate a model on many parameter sets
    and optionally add Gaussian shot noise.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = self.evaluate(observables, parameter_sets)
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridQLSTM", "FastEstimator"]
