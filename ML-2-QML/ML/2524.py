"""Hybrid quantum‑classical classifier integrating QCNN and variational layers."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Tuple

# Import the classical QCNN model from the seed
from.QCNN import QCNNModel


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[object, Iterable, Iterable, List]:
    """
    Construct a quantum circuit that first encodes data, then applies
    QCNN‑style convolution and pooling layers, followed by a variational depth
    and measurement observables.  The return signature matches the original
    seed so existing code remains functional.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers after the QCNN blocks.

    Returns
    -------
    circuit : QuantumCircuit
        The assembled quantum circuit.
    encoding : Iterable
        Parameters used for data encoding.
    weights : Iterable
        Variational parameters.
    observables : List[SparsePauliOp]
        Observables for expectation value extraction.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    # ---------- Data encoding ----------
    encoding = ParameterVector("x", num_qubits)
    circuit = QuantumCircuit(num_qubits)
    for qubit, param in zip(range(num_qubits), encoding):
        circuit.rx(param, qubit)

    # ---------- QCNN convolution block ----------
    def conv_circuit(params: ParameterVector, qubits: List[int]) -> QuantumCircuit:
        """Two‑qubit convolution unitary used in QCNN."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        return sub

    conv_params = ParameterVector("c", length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits - 1, 2):
        sub = conv_circuit(conv_params[i // 2 * 3 : i // 2 * 3 + 3], [i, i + 1])
        circuit.append(sub.to_instruction(), [i, i + 1])

    # ---------- QCNN pooling block ----------
    def pool_circuit(params: ParameterVector, qubits: List[int]) -> QuantumCircuit:
        """Two‑qubit pooling unitary used in QCNN."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        return sub

    pool_params = ParameterVector("p", length=(num_qubits // 2) * 3)
    for i in range(0, num_qubits - 1, 2):
        sub = pool_circuit(pool_params[i // 2 * 3 : i // 2 * 3 + 3], [i, i + 1])
        circuit.append(sub.to_instruction(), [i, i + 1])

    # ---------- Variational depth ----------
    weights = ParameterVector("theta", length=num_qubits * depth)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # ---------- Observables ----------
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


class HybridQuantumCNNClassifier(nn.Module):
    """
    Hybrid model that extracts features with a classical QCNN and maps the
    reduced representation to a quantum circuit for classification.
    """

    def __init__(
        self,
        num_features: int,
        num_qubits: int = 4,
        depth: int = 2,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.qcnn = QCNNModel()
        # Map QCNN output (scalar) to a vector suitable for quantum encoding
        self.fc = nn.Linear(1, num_qubits)
        self.num_qubits = num_qubits
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical QCNN feature extraction
        feat = self.qcnn(x).view(-1)
        # Linear mapping to qubit‑dimension
        qubit_vals = self.fc(feat)
        # In a full hybrid implementation, qubit_vals would be fed into the
        # quantum circuit and expectation values returned.  Here we simply
        # expose the classical logits for downstream training.
        return qubit_vals


__all__ = ["HybridQuantumCNNClassifier", "build_classifier_circuit"]
