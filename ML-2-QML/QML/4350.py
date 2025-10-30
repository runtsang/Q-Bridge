"""Hybrid quantum classifier that encodes features, applies self‑attention and graph‑entanglement."""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import networkx as nx
import scipy as sc

# --------------------------------------------------------------------------- #
#  Utility: fidelity‑based adjacency matrix
# --------------------------------------------------------------------------- #
def _fidelity_adjacency_matrix(
    states: Iterable[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> np.ndarray:
    n = len(states)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            fid = np.abs(states[i].conj().dot(states[j])) ** 2
            if fid >= threshold:
                adj[i, j] = adj[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                adj[i, j] = adj[j, i] = secondary_weight
    return adj

# --------------------------------------------------------------------------- #
#  Quantum self‑attention subcircuit
# --------------------------------------------------------------------------- #
class _QuantumSelfAttention:
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qreg = QuantumRegister(n_qubits)
        self.creg = ClassicalRegister(n_qubits)

    def circuit(
        self,
        rot_params: np.ndarray,
        ent_params: np.ndarray,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qreg, self.creg)
        for i in range(self.n_qubits):
            qc.rx(rot_params[3 * i], self.qreg[i])
            qc.ry(rot_params[3 * i + 1], self.qreg[i])
            qc.rz(rot_params[3 * i + 2], self.qreg[i])
        for i in range(self.n_qubits - 1):
            qc.crx(ent_params[i], self.qreg[i], self.qreg[i + 1])
        qc.measure(self.qreg, self.creg)
        return qc

# --------------------------------------------------------------------------- #
#  Hybrid quantum classifier – public API
# --------------------------------------------------------------------------- #
class HybridClassifier:
    """
    Quantum hybrid classifier that encodes classical features into qubits,
    applies a self‑attention subcircuit, entangles qubits according to a
    fidelity‑based adjacency, and measures Pauli‑Z observables.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        embed_dim: int,
        graph_arch: List[int],
        prototype_states: List[np.ndarray],
        fidelity_threshold: float,
        *,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.embed_dim = embed_dim
        self.graph_arch = graph_arch
        self.prototype_states = prototype_states
        self.fidelity_threshold = fidelity_threshold
        self.secondary_threshold = secondary_threshold
        self.secondary_weight = secondary_weight

        # Parameter vectors
        self.encoding_params = ParameterVector("enc", num_qubits)
        self.attn_rot = ParameterVector("rot", 3 * num_qubits)
        self.attn_ent = ParameterVector("ent", num_qubits - 1)
        self.weights = ParameterVector("w", num_qubits * depth)

        # Base encoding
        self.base_circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            self.base_circuit.rx(self.encoding_params[i], i)

        # Self‑attention subcircuit
        self.attn_sub = _QuantumSelfAttention(num_qubits)

        # Fidelity‑based adjacency for entanglement
        self.adj_matrix = _fidelity_adjacency_matrix(
            prototype_states,
            fidelity_threshold,
            secondary=secondary_threshold,
            secondary_weight=secondary_weight,
        )

        # Observables
        self.observables = [
            SparsePauliOp("Z" * i + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
        ]

    def build_circuit(self) -> QuantumCircuit:
        qc = self.base_circuit.copy()
        # Self‑attention block
        qc += self.attn_sub.circuit(self.attn_rot, self.attn_ent)
        # Variational layers with graph‑based entanglement
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(self.weights[idx], q)
                idx += 1
            for i in range(self.num_qubits - 1):
                if self.adj_matrix[i, i + 1] > 0:
                    qc.cz(i, i + 1)
        return qc

    def run(
        self,
        backend,
        feature_vector: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        qc = self.build_circuit()
        # Bind encoding parameters to the feature vector
        param_dict = {self.encoding_params[i]: feature_vector[i] for i in range(self.num_qubits)}
        qc = qc.bind_parameters(param_dict)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)

# --------------------------------------------------------------------------- #
#  Function that mimics the classical interface
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    embed_dim: int,
    graph_arch: List[int],
    prototype_states: List[np.ndarray],
    fidelity_threshold: float,
    *,
    secondary_threshold: float | None = None,
    secondary_weight: float = 0.5,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Return a quantum circuit together with metadata that mirrors the classical
    interface: encoding vector, variational parameters and observables.
    """
    classifier = HybridClassifier(
        num_qubits=num_qubits,
        depth=depth,
        embed_dim=embed_dim,
        graph_arch=graph_arch,
        prototype_states=prototype_states,
        fidelity_threshold=fidelity_threshold,
        secondary_threshold=secondary_threshold,
        secondary_weight=secondary_weight,
    )
    circuit = classifier.build_circuit()
    encoding = list(classifier.encoding_params)
    weights = list(classifier.weights)
    observables = classifier.observables
    return circuit, encoding, weights, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
