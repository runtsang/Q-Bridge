"""Enhanced quanvolution classifier – quantum implementation.

The quantum module re‑implements the classical filter using a
parameterised quantum kernel and optionally a quantum LSTM cell
from the QLSTM seed.  It keeps the same public interface as the
classical version so that downstream code can switch between
modes without modification.

Key quantum ingredients
-----------------------
* :class:`EnhancedQuanvolutionFilter` – a 2‑qubit quantum kernel applied
  to every 2×2 patch of the input image.
* :class:`EnhancedQLSTM` – a quantum LSTM cell that uses small
  variational circuits for each gate.
* :class:`FastEstimator` – wrapper around the QML FastBaseEstimator
  that evaluates expectation values for a list of parameter sets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

# --------------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: tq.QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> tq.QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[tq.QuantumOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = tq.Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""
    def evaluate(self, observables: Iterable[tq.QuantumOperator], parameter_sets: Sequence[Sequence[float]], *, shots: int | None = None, seed: int | None = None) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(value.real, 1 / shots) + 1j * rng.normal(value.imag, 1 / shots) for value in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#  Quantum quanvolution filter
# --------------------------------------------------------------------------- #

class EnhancedQuanvolutionFilter(tq.QuantumModule):
    """Apply a 2‑qubit quantum kernel to each 2×2 patch of the input image."""
    def __init__(self, n_wires: int = 4, n_ops: int = 10) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
#  Quantum LSTM cell
# --------------------------------------------------------------------------- #

class EnhancedQLSTM(tq.QuantumModule):
    """Quantum LSTM cell where each gate is a small variational circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


# --------------------------------------------------------------------------- #
#  Main quantum classifier
# --------------------------------------------------------------------------- #

class EnhancedQuanvolutionClassifier(nn.Module):
    """
    Quantum-enhanced quanvolution classifier.

    Parameters
    ----------
    use_lstm : bool, default False
        If True, a quantum LSTM processes the flattened features before the
        final linear head.  The LSTM uses the same number of qubits as the
        quanvolution filter for consistency.
    """
    def __init__(self, use_lstm: bool = False) -> None:
        super().__init__()
        self.use_lstm = use_lstm
        self.qfilter = EnhancedQuanvolutionFilter()
        feature_dim = 4 * 14 * 14  # matches the classical counterpart

        if self.use_lstm:
            self.lstm = EnhancedQLSTM(input_dim=feature_dim, hidden_dim=feature_dim // 2, n_qubits=self.qfilter.n_wires)
            lstm_out_dim = feature_dim
        else:
            lstm_out_dim = feature_dim

        self.classifier = nn.Linear(lstm_out_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        features = self.qfilter(x)  # (batch, feature_dim)

        if self.use_lstm:
            lstm_out, _ = self.lstm(features.unsqueeze(1))
            features = lstm_out.squeeze(1)

        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["EnhancedQuanvolutionClassifier", "FastEstimator", "FastBaseEstimator"]
