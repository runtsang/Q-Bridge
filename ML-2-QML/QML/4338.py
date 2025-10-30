"""Quantum implementation of the hybrid estimator.

Provides quantum counterparts for convolution, LSTM and self‑attention
as well as an estimator that evaluates parameterised circuits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List

# Quantum primitives ---------------------------------------------------------

class ConvCircuit:
    """Quantum convolutional filter for 2×2 kernels."""
    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.base_circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self.base_circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.base_circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = data.reshape(self.n_qubits)
        circ = self.base_circuit.copy()
        for i, val in enumerate(data):
            angle = np.pi if val > self.threshold else 0.0
            circ.rx(angle, i)
        job = execute(circ, self.backend, shots=self.shots)
        result = job.result().get_counts(circ)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class QuantumLSTM:
    """Quantum‑enhanced LSTM cell implemented with small parameterised circuits."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.gates = {
            "forget": self._gate_circuit(),
            "input": self._gate_circuit(),
            "update": self._gate_circuit(),
            "output": self._gate_circuit(),
        }
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _gate_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(qiskit.circuit.Parameter(f"theta{i}"), i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def _run_gate(self, gate: QuantumCircuit, params: np.ndarray) -> float:
        bind = {gate.parameters[i]: params[i] for i in range(gate.num_parameters)}
        job = execute(gate, self.backend, shots=self.shots, parameter_binds=[bind])
        result = job.result().get_counts(gate)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    def forward(
        self,
        inputs: np.ndarray,
        states: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs:
            combined = np.concatenate([x, hx])
            f = self._run_gate(self.gates["forget"], combined)
            i = self._run_gate(self.gates["input"], combined)
            g = self._run_gate(self.gates["update"], combined)
            o = self._run_gate(self.gates["output"], combined)
            cx = f * cx + i * g
            hx = o * np.tanh(cx)
            outputs.append(hx)
        return np.stack(outputs), (hx, cx)

    def _init_states(
        self,
        inputs: np.ndarray,
        states: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if states is not None:
            return states
        batch_size = inputs.shape[0]
        return (
            np.zeros((batch_size, self.hidden_dim)),
            np.zeros((batch_size, self.hidden_dim)),
        )


class QuantumSelfAttention:
    """Quantum self‑attention block using rotation and controlled‑X gates."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int | None = None,
    ) -> dict[str, int]:
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=shots or self.shots)
        return job.result().get_counts(qc)


# Estimator ---------------------------------------------------------------

class FastBaseEstimator:
    """Hybrid estimator that evaluates a parameterised QuantumCircuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional shot‑noise simulation to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex]]:
        if shots is None:
            return super().evaluate(observables, parameter_sets)
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            job = execute(circ, Aer.get_backend("qasm_simulator"), shots=shots)
            result = job.result()
            row = []
            for obs in observables:
                counts = result.get_counts(circ)
                exp = sum(int(bit) for key in counts for bit in key) / (shots * circ.num_qubits)
                row.append(complex(exp))
            results.append(row)
        return results


__all__ = [
    "ConvCircuit",
    "QuantumLSTM",
    "QuantumSelfAttention",
    "FastBaseEstimator",
    "FastEstimator",
]
