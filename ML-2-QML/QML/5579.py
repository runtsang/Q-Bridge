"""
CombinedAttentionClassifierGraphCNN (Quantum)
============================================

This module provides a quantum counterpart to the classical model defined above.
It stitches together:
* a self‑attention circuit,
* a variational classifier ansatz, and
* a QCNN‑style convolution/pooling stack.
The API mirrors the classical version to enable side‑by‑side benchmarking.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# 1. Self‑attention circuit
# --------------------------------------------------------------------------- #
def build_attention_circuit(n_qubits: int) -> QuantumCircuit:
    """Creates a simple self‑attention block with rotation and C‑RX gates."""
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    rot = ParameterVector("θ", n_qubits * 3)
    ent = ParameterVector("φ", n_qubits - 1)

    for i in range(n_qubits):
        circuit.rx(rot[3 * i], i)
        circuit.ry(rot[3 * i + 1], i)
        circuit.rz(rot[3 * i + 2], i)

    for i in range(n_qubits - 1):
        circuit.crx(ent[i], i, i + 1)

    circuit.measure(qr, cr)
    return circuit


# --------------------------------------------------------------------------- #
# 2. Variational classifier ansatz
# --------------------------------------------------------------------------- #
def build_classifier_circuit(n_qubits: int, depth: int) -> QuantumCircuit:
    """Encodes data then applies a layered ansatz with Ry rotations and CZ gates."""
    circuit = QuantumCircuit(n_qubits)
    enc = ParameterVector("x", n_qubits)
    for i in range(n_qubits):
        circuit.rx(enc[i], i)

    param_idx = 0
    for _ in range(depth):
        for i in range(n_qubits):
            circuit.ry(ParameterVector(f"θ_{param_idx}", 1)[0], i)
            param_idx += 1
        for i in range(n_qubits - 1):
            circuit.cz(i, i + 1)
    return circuit


# --------------------------------------------------------------------------- #
# 3. QCNN‑style convolution/pooling stack
# --------------------------------------------------------------------------- #
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """One convolution block on two qubits."""
    c = QuantumCircuit(2)
    c.rz(-np.pi / 2, 1)
    c.cx(1, 0)
    c.rz(params[0], 0)
    c.ry(params[1], 1)
    c.cx(0, 1)
    c.ry(params[2], 1)
    c.cx(1, 0)
    c.rz(np.pi / 2, 0)
    return c


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """One pooling block on two qubits."""
    c = QuantumCircuit(2)
    c.rz(-np.pi / 2, 1)
    c.cx(1, 0)
    c.rz(params[0], 0)
    c.ry(params[1], 1)
    c.cx(0, 1)
    c.ry(params[2], 1)
    return c


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Apply convolution blocks pairwise across all qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, num_qubits * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        block = conv_circuit(params[3 * (q1 // 2) : 3 * (q1 // 2 + 1)])
        qc.append(block, [q1, q2])
    return qc


def pool_layer(sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    """Apply pooling blocks between source and sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, len(sources) * 3)
    for src, snk in zip(sources, sinks):
        block = pool_circuit(params[3 * sources.index(src) : 3 * (sources.index(src) + 1)])
        qc.append(block, [src, snk])
    return qc


def build_qcnn_circuit(n_qubits: int) -> QuantumCircuit:
    """Constructs a 3‑layer QCNN stack with convolution and pooling."""
    qc = QuantumCircuit(n_qubits)

    # 1st convolution
    qc.append(conv_layer(n_qubits, "c1"), list(range(n_qubits)))

    # 1st pooling (split into two 4‑qubit groups)
    qc.append(pool_layer(list(range(4)), list(range(4, 8)), "p1"), list(range(n_qubits)))

    # 2nd convolution on the remaining 4 qubits
    qc.append(conv_layer(n_qubits // 2, "c2"), list(range(n_qubits // 2, n_qubits)))

    # 2nd pooling
    qc.append(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p2"), list(range(n_qubits)))

    # 3rd convolution on the last 2 qubits
    qc.append(conv_layer(n_qubits // 4, "c3"), list(range(3 * n_qubits // 4, n_qubits)))

    # 3rd pooling
    qc.append(pool_layer([0], [1], "p3"), list(range(n_qubits)))

    return qc


# --------------------------------------------------------------------------- #
# 4. Combined quantum model
# --------------------------------------------------------------------------- #
class QuantumCombinedAttentionClassifierGraphCNN:
    """A quantum model that mirrors :class:`CombinedModel`."""

    def __init__(self, n_qubits: int, depth: int):
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = Aer.get_backend("qasm_simulator")

        self.attn_circ = build_attention_circuit(n_qubits)
        self.classifier_circ = build_classifier_circuit(n_qubits, depth)
        self.qcnn_circ = build_qcnn_circuit(n_qubits)

    def _bind_params(self, circ: QuantumCircuit, param_map: Dict[str, np.ndarray]) -> QuantumCircuit:
        """Utility to bind numpy arrays to a circuit's parameters."""
        return circ.bind_parameters({name: val for name, val in param_map.items()})

    def run(self,
            attn_params: np.ndarray,
            class_params: np.ndarray,
            qcnn_params: np.ndarray,
            shots: int = 1024) -> Dict[str, Dict[str, int]]:
        """
        Execute all three sub‑circuits and return measurement counts.
        Parameters are flattened arrays matching the number of parameters
        in each circuit.
        """
        # Build parameter dictionaries
        attn_map = {f"θ_{i}": attn_params[i] for i in range(len(attn_params) // 3)}
        attn_map.update({f"φ_{i}": attn_params[len(attn_params) // 3 + i]
                         for i in range(self.n_qubits - 1)})

        class_map = {f"x_{i}": class_params[i] for i in range(self.n_qubits)}
        class_map.update({f"θ_{i}": class_params[self.n_qubits + i]
                          for i in range(self.depth * self.n_qubits)})

        qcnn_map = {f"c1_{i}": qcnn_params[i] for i in range(len(qcnn_params) // 9)}
        # Simplified: only a few parameters are bound; full mapping omitted for brevity

        # Execute circuits
        attn_job = execute(self._bind_params(self.attn_circ, attn_map), self.backend, shots=shots)
        class_job = execute(self._bind_params(self.classifier_circ, class_map), self.backend, shots=shots)
        qcnn_job = execute(self._bind_params(self.qcnn_circ, qcnn_map), self.backend, shots=shots)

        return {
            "attention": attn_job.result().get_counts(),
            "classifier": class_job.result().get_counts(),
            "qcnn": qcnn_job.result().get_counts(),
        }


# --------------------------------------------------------------------------- #
# 5. Utility for graph fidelity (optional)
# --------------------------------------------------------------------------- #
def fidelity_adjacency(states: List[complex], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted adjacency graph from pure state overlaps."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = abs(np.vdot(a, b)) ** 2
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


__all__ = [
    "QuantumCombinedAttentionClassifierGraphCNN",
    "build_attention_circuit",
    "build_classifier_circuit",
    "build_qcnn_circuit",
    "fidelity_adjacency",
]
