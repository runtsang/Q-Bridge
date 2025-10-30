"""Quantum‑classical hybrid classifier that uses a quantum self‑attention subcircuit
followed by a variational classification ansatz.  The implementation extends
the original incremental data‑uploading circuit by inserting a self‑attention
block, thereby providing a richer feature map."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]

class HybridQuantumClassifier:
    """Hybrid quantum classifier that first applies a quantum self‑attention
    subcircuit, then a variational classification ansatz."""
    def __init__(self, num_qubits: int, depth: int, embed_dim: int = 4,
                 backend=None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.embed_dim = embed_dim
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def _build_self_attention_circuit(self,
                                      rotation_params: ParameterVector,
                                      entangle_params: ParameterVector) -> QuantumCircuit:
        """Quantum self‑attention block."""
        qc = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.num_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        return qc

    def _build_classifier_ansatz(self,
                                 weights: ParameterVector) -> QuantumCircuit:
        """Variational classification ansatz."""
        qc = QuantumCircuit(self.num_qubits)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        return qc

    def build_full_circuit(self,
                           rotation_params: ParameterVector,
                           entangle_params: ParameterVector,
                           weights: ParameterVector) -> QuantumCircuit:
        """Return a circuit that concatenates the self‑attention block and
        the classification ansatz."""
        qc = QuantumCircuit(self.num_qubits)
        sa_circ = self._build_self_attention_circuit(rotation_params, entangle_params)
        qc.append(sa_circ.to_instruction(), qc.qubits)
        clf_circ = self._build_classifier_ansatz(weights)
        qc.append(clf_circ.to_instruction(), qc.qubits)
        return qc

    def run(self,
            rotation_params_vals: np.ndarray,
            entangle_params_vals: np.ndarray,
            weights_vals: np.ndarray,
            shots: int = 1024) -> dict:
        """Execute the full circuit with the supplied parameter values."""
        rotation_params = ParameterVector("x", self.num_qubits * 3)
        entangle_params = ParameterVector("t", self.num_qubits - 1)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        circuit = self.build_full_circuit(rotation_params, entangle_params, weights)
        param_map = {}
        param_map.update({p: v for p, v in zip(rotation_params,
                                               rotation_params_vals)})
        param_map.update({p: v for p, v in zip(entangle_params,
                                               entangle_params_vals)})
        param_map.update({p: v for p, v in zip(weights,
                                               weights_vals)})
        bound_circuit = circuit.bind_parameters(param_map)
        job = qiskit.execute(bound_circuit, self.backend, shots=shots)
        return job.result().get_counts(bound_circuit)

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple:
    """Return a tuple (circuit, encoding, weight_sizes, observables) compatible
    with the original anchor API."""
    rotation_params = ParameterVector("x", num_qubits * 3)
    entangle_params = ParameterVector("t", num_qubits - 1)
    weights = ParameterVector("theta", num_qubits * depth)
    qc = QuantumCircuit(num_qubits)
    # Self‑attention block
    for i in range(num_qubits):
        qc.rx(rotation_params[3 * i], i)
        qc.ry(rotation_params[3 * i + 1], i)
        qc.rz(rotation_params[3 * i + 2], i)
    for i in range(num_qubits - 1):
        qc.crx(entangle_params[i], i, i + 1)
    # Classification ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)
    # Observables
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    encoding = [rotation_params, entangle_params, weights]
    weight_sizes = [len(rotation_params) + len(entangle_params), len(weights)]
    return qc, encoding, weight_sizes, observables
