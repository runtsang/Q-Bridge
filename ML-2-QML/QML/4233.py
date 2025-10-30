"""Quantum hybrid self‑attention implementation.

The quantum counterpart of `HybridSelfAttention` builds a parametrised
circuit that mimics the attention mechanism via rotation and entanglement
gates.  It integrates with the lightweight estimator utilities and the
synthetic regression dataset defined in the classical module.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ---- Lightweight estimator utilities (FastBaseEstimator) ----
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, values: list[float]) -> QuantumCircuit:
        if len(values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch")
        mapping = dict(zip(self.parameters, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: list[BaseOperator], parameter_sets: list[list[float]]) -> list[list[complex]]:
        results: list[list[complex]] = []
        for params in parameter_sets:
            bound = self._bind(params)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ---- Synthetic regression dataset (QuantumRegression) ----
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset:
    """Dataset that returns a state vector and a regression target."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return {
            "states": self.states[idx],
            "target": self.labels[idx],
        }

# ---- Quantum self‑attention model ----
class HybridSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""

    def __init__(self, n_qubits: int, mode: str = "quantum"):
        self.n_qubits = n_qubits
        self.mode = mode
        self.qreg = QuantumRegister(n_qubits, "q")
        self.creg = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qreg, self.creg)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qreg, self.creg)
        # Rotation layer
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        # Entanglement layer (CRX)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qreg, self.creg)
        return circ

    def run(self, backend: qiskit.providers.Backend, rotation_params: np.ndarray,
            entangle_params: np.ndarray, shots: int = 1024) -> dict[str, int]:
        circ = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circ, backend, shots=shots)
        return job.result().get_counts(circ)

__all__ = ["HybridSelfAttention", "RegressionDataset", "FastBaseEstimator"]
