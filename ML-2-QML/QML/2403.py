"""Hybrid quantum estimator that augments a Qiskit EstimatorQNN with a parameterised self‑attention circuit."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

# Quantum self‑attention helper
class QuantumSelfAttention:
    """Quantum circuit implementing a self‑attention style block."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


def HybridEstimatorQNN():
    """Return a hybrid quantum estimator with an embedded self‑attention circuit."""

    # Build the base quantum regression circuit
    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)

    # Self‑attention parameters
    attn_params = np.random.randn(12)  # 4 qubits * 3 rotation params
    entangle_params = np.random.randn(3)  # 3 entangling gates

    # Build attention circuit
    attention = QuantumSelfAttention(n_qubits=4)
    attention_circuit = attention._build_circuit(attn_params, entangle_params)

    # Combine circuits
    full_circuit = qc1.compose(attention_circuit)

    # Observable
    observable1 = SparsePauliOp.from_list([("Y" * full_circuit.num_qubits, 1)])

    # Estimator
    estimator = StatevectorEstimator()
    estimator_qnn = EstimatorQNN(
        circuit=full_circuit,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )
    return estimator_qnn


__all__ = ["HybridEstimatorQNN"]
