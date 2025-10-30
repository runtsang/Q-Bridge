"""Hybrid LSTM implementation with quantum gates and shot‑noise estimation.

The `HybridQLSTM` class implements a sequence‑to‑sequence LSTM where each gate
is realised by a small parameterised quantum circuit.  The circuits are
constructed with torchquantum and employ RX rotations followed by a chain
of CNOT gates.  Outputs are obtained by measuring all qubits in the Pauli‑Z
basis.  An optional Gaussian noise model mimics finite‑shot statistics.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridQLSTM(nn.Module):
    """Quantum LSTM where gates are realised by small quantum circuits.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dim : int
        Dimensionality of hidden state.
    n_qubits : int
        Number of qubits used in each gate circuit.
    dropout : float, default=0.0
        Dropout probability applied to the output of each cell.
    layer_norm : bool, default=False
        Whether to apply layer normalisation after each gate.
    """

    class QLayer(tq.QuantumModule):
        """Parameterised quantum layer used for each LSTM gate."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, n_wires)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = dropout
        self.layer_norm = layer_norm

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout_layer(self.ln(hx))
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def evaluate_shots(
        self,
        inputs: torch.Tensor,
        shots: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Return outputs with Gaussian noise mimicking finite‑shot statistics."""
        outputs, _ = self.forward(inputs)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1 / np.sqrt(shots), size=outputs.shape)
        return outputs + torch.tensor(noise, device=outputs.device, dtype=outputs.dtype)

class FastEstimator:
    """Estimator for the quantum hybrid LSTM that can add Gaussian shot noise.

    This class mirrors the classical `FastEstimator` but operates on the
    quantum model and allows specifying a shot‑count to simulate measurement
    noise.
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
