"""Quantum‑enhanced transformer with a hybrid estimator that can be used as a sub‑module in a Qiskit circuit."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, List

import torch
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import BaseOperator


class _QuantumBaseEstimator:
    """Internal base for quantum estimator."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.params, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        if shots is None:
            return self._evaluate_exact(observables, parameter_sets)
        else:
            return self._evaluate_shot(observables, parameter_sets, shots, seed)

    def _evaluate_exact(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        results: List[List[float]] = []
        for vals in parameter_sets:
            bound = self._bind(vals)
            state = Statevector.from_instruction(bound)
            row = [float(state.expectation_value(obs)) for obs in observables]
            results.append(row)
        return results

    def _evaluate_shot(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        rng = np.random.default_rng(seed)
        results: List[List[float]] = []
        for vals in parameter_sets:
            bound = self._bind(vals)
            bound = transpile(bound, optimization_level=0)
            state = Statevector.from_instruction(bound)
            exact = [float(state.expectation_value(obs)) for obs in observables]
            noisy = rng.normal(loc=exact, scale=1 / shots)
            results.append(noisy.tolist())
        return results


class FastHybridEstimator(_QuantumBaseEstimator):
    """Quantum estimator that can be used inside a hybrid pipeline."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        super().__init__(circuit)


# --------------------------------------------------------------------------- #
#  Quantum‑enhanced transformer block definitions (minimal viable implementation)
# --------------------------------------------------------------------------- #

class QuantumTransformer:
    """
    Builds a simple variational quantum circuit that can be embedded in a Qiskit workflow.
    The circuit consists of single‑qubit rotations followed by a depth of entangling layers.
    """

    def __init__(self, n_qubits: int, depth: int = 1) -> None:
        self.n_qubits = n_qubits
        self.depth = depth

    def build_circuit(self, param_dict: dict[str, float]) -> QuantumCircuit:
        """Return a parameterised Qiskit circuit with the given parameters."""
        qreg = QuantumRegister(self.n_qubits)
        creg = ClassicalRegister(self.n_qubits)
        circuit = QuantumCircuit(qreg, creg)

        # Single‑qubit rotations
        for i in range(self.n_qubits):
            theta = Parameter(f"theta_{i}")
            circuit.ry(theta, i)

        # Entangling layers
        for _ in range(self.depth):
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.cx(self.n_qubits - 1, 0)

        # Bind parameters
        bound = circuit.bind_parameters(param_dict)
        return bound


class QuantumAttention(tq.QuantumModule):
    """
    Simple quantum attention module that applies a parameterised rotation
    to each head and measures in the Z basis.
    """

    def __init__(self, n_wires: int, n_heads: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_heads = n_heads
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [idx], "func": "rx", "wires": [idx]}
                for idx in range(n_wires)
            ]
        )
        self.parameters = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)


class QuantumFeedForward(tq.QuantumModule):
    """
    Quantum feed‑forward block that maps input amplitudes to a higher‑dimensional space
    via parameterised rotations and then projects back.
    """

    def __init__(self, n_wires: int, output_dim: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.output_dim = output_dim
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [idx], "func": "ry", "wires": [idx]}
                for idx in range(n_wires)
            ]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)


__all__ = [
    "FastHybridEstimator",
    "QuantumTransformer",
    "QuantumAttention",
    "QuantumFeedForward",
]
