"""Hybrid quantum self‑attention + QCNN using Qiskit.

The circuit is built by concatenating a parameterised self‑attention block
with a QCNN ansatz that mirrors the structure from the classical
reference.  Parameters are grouped into three sets:

* attention_params: array of shape (3 * n_qubits,)
* conv_params / pool_params: iterable of flattened parameter vectors for each
  convolution/pooling layer.  The ordering follows the build order.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator as QiskitEstimator
from typing import Sequence, Dict

# --- Self‑Attention Circuit -----------------------------------------------
def _attention_circuit(n_qubits: int, rotation_params: ParameterVector, entangle_params: ParameterVector) -> QuantumCircuit:
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    circuit = QuantumCircuit(qr, cr)
    for i in range(n_qubits):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)
    for i in range(n_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)
    circuit.measure(qr, cr)
    return circuit

# --- QCNN Ansätze -----------------------------------------------
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _conv_circuit(params[3 * (q1 // 2) : 3 * (q1 // 2 + 1)])
        qc.append(sub.to_instruction(), [q1, q2])
    return qc

def _pool_layer(sources: Sequence[int], sinks: Sequence[int], prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=num_qubits // 2 * 3)
    for idx, (src, sink) in enumerate(zip(sources, sinks)):
        sub = _pool_circuit(params[3 * idx : 3 * idx + 3])
        qc.append(sub.to_instruction(), [src, sink])
    return qc

# --- Full QCNN Circuit -----------------------------------------------
def _build_qcnn_circuit(num_qubits: int = 8) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(_conv_layer(num_qubits, "c1"), inplace=True)
    circuit.compose(_pool_layer(list(range(4)), list(range(4, 8)), "p1"), inplace=True)
    circuit.compose(_conv_layer(4, "c2"), inplace=True)
    circuit.compose(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p2"), inplace=True)
    circuit.compose(_conv_layer(2, "c3"), inplace=True)
    circuit.compose(_pool_layer([0], [1], "p3"), inplace=True)
    return circuit

# --- Hybrid Quantum Class -----------------------------------------------
class HybridQuantumSelfAttentionQCNN:
    """
    Hybrid circuit that first applies a self‑attention block followed
    by a QCNN ansatz.  Parameters are bound at runtime via a
    ParameterVector mapping.  The circuit is evaluated with a
    Qiskit Estimator backend.
    """
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(n_qubits)
        # Parameter vectors for attention
        self.rotation_params = ParameterVector("rot", length=3 * n_qubits)
        self.entangle_params = ParameterVector("ent", length=n_qubits - 1)

        # Sub‑circuits
        attention_sub = _attention_circuit(n_qubits, self.rotation_params, self.entangle_params)
        qcnn_sub = _build_qcnn_circuit()

        # Compose full circuit
        self.circuit.compose(attention_sub, inplace=True)
        self.circuit.compose(qcnn_sub, inplace=True)

    def run(
        self,
        attention_params: np.ndarray,
        conv_params: Sequence[Sequence[float]],
        pool_params: Sequence[Sequence[float]],
        shots: int = 1024,
    ) -> Dict[str, int]:
        """
        Execute the hybrid circuit with the supplied parameter sets.
        """
        flat_qcnn = []
        for layer in conv_params + pool_params:
            flat_qcnn.extend(layer)
        all_params = list(self.circuit.parameters)
        if len(all_params)!= len(attention_params) + len(flat_qcnn):
            raise ValueError("Parameter count mismatch.")
        mapping = {p: v for p, v in zip(all_params, np.concatenate([attention_params, flat_qcnn]))}
        bound = self.circuit.assign_parameters(mapping, inplace=False)
        job = self.backend.run(bound, shots=shots)
        return job.result().get_counts(bound)

# --- Factory ---------------------------------------------------------------
def HybridQuantumSelfAttentionQCNNFactory() -> HybridQuantumSelfAttentionQCNN:
    return HybridQuantumSelfAttentionQCNN()

__all__ = [
    "HybridQuantumSelfAttentionQCNN",
    "HybridQuantumSelfAttentionQCNNFactory",
]
