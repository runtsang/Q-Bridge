"""Quantum LSTM implementation and Qiskit estimator.

This module implements a quantum LSTM cell using torchquantum and
provides a minimal Qiskit-based estimator that can be used by
:class:`UnifiedBaseEstimator` in the ML side.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Optional dependency: torchquantum
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torchquantum is required for the quantum LSTM implementation."
    ) from exc

# --------------------------------------------------------------------------- #
# Quantum LSTM
# --------------------------------------------------------------------------- #
class QLSTMQuantum(nn.Module):
    """Quantum LSTM cell where each gate is realised by a small PQC."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Simple encoder that maps classical inputs to qubit rotations
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            # Parameterised gates for each wire
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            # Simple entangling pattern
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Each gate has its own PQC
        self.forget = self._QLayer(n_qubits)
        self.input = self._QLayer(n_qubits)
        self.update = self._QLayer(n_qubits)
        self.output = self._QLayer(n_qubits)

        # Linear projections to qubit count
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

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
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

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

# --------------------------------------------------------------------------- #
# Qiskit-based estimator
# --------------------------------------------------------------------------- #
from qiskit import Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QBaseEstimator:
    """Evaluate expectation values of observables for a parametrised Qiskit circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["QLSTMQuantum", "QBaseEstimator"]
