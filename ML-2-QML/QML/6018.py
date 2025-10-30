"""Quantum hybrid QCNN implementation using Qiskit.

The circuit is built from the adjacency matrix produced by the
classical FeatureGraphNet.  Each node of the graph corresponds to a qubit.
The adjacency determines which qubits are entangled during the pooling
layers.  Convolution layers are applied uniformly across all qubits.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def conv_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """
    Two‑qubit convolution unitary used in the QCNN seed.
    """
    qc = QuantumCircuit(len(qubits))
    q0, q1 = qubits[0], qubits[1]
    qc.rz(-np.pi / 2, q1)
    qc.cx(q1, q0)
    qc.rz(params[0], q0)
    qc.ry(params[1], q1)
    qc.cx(q0, q1)
    qc.ry(params[2], q1)
    qc.cx(q1, q0)
    qc.rz(np.pi / 2, q0)
    return qc

def pool_circuit(params: ParameterVector, qubits: list[int]) -> QuantumCircuit:
    """
    Two‑qubit pooling unitary used in the QCNN seed.
    """
    qc = QuantumCircuit(len(qubits))
    q0, q1 = qubits[0], qubits[1]
    qc.rz(-np.pi / 2, q1)
    qc.cx(q1, q0)
    qc.rz(params[0], q0)
    qc.ry(params[1], q1)
    qc.cx(q0, q1)
    qc.ry(params[2], q1)
    return qc

def build_qcnn(adj_matrix: np.ndarray) -> EstimatorQNN:
    """
    Build a variational QCNN whose connectivity is dictated by *adj_matrix*.
    *adj_matrix* should be a 2‑D array of shape (n, n) where n is the number
    of qubits.  Edges with weight above a threshold are treated as
    entanglement links for pooling.
    """
    n_qubits = adj_matrix.shape[0]
    feature_map = ZFeatureMap(num_qubits=n_qubits, reps=1, entanglement='full')
    estimator = Estimator()

    # Determine pooling pairs from adjacency
    threshold = 0.2
    pooling_pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
                     if adj_matrix[i, j] > threshold]

    # Ansatz: convolution → pooling → convolution → pooling →...
    circuit = QuantumCircuit(n_qubits)
    # Convolution 1
    params_c1 = ParameterVector("c1", length=3 * n_qubits)
    for i in range(n_qubits):
        sub = conv_circuit(params_c1[3 * i: 3 * (i + 1)], [i, (i + 1) % n_qubits])
        circuit.append(sub.to_instruction(), [i, (i + 1) % n_qubits])
    # Pooling 1
    params_p1 = ParameterVector("p1", length=3 * len(pooling_pairs))
    for idx, (i, j) in enumerate(pooling_pairs):
        sub = pool_circuit(params_p1[3 * idx: 3 * (idx + 1)], [i, j])
        circuit.append(sub.to_instruction(), [i, j])

    # Convolution 2
    params_c2 = ParameterVector("c2", length=3 * n_qubits)
    for i in range(n_qubits):
        sub = conv_circuit(params_c2[3 * i: 3 * (i + 1)], [i, (i + 1) % n_qubits])
        circuit.append(sub.to_instruction(), [i, (i + 1) % n_qubits])

    # Pooling 2
    params_p2 = ParameterVector("p2", length=3 * len(pooling_pairs))
    for idx, (i, j) in enumerate(pooling_pairs):
        sub = pool_circuit(params_p2[3 * idx: 3 * (idx + 1)], [i, j])
        circuit.append(sub.to_instruction(), [i, j])

    # Final observable
    observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1.0)])

    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=list(params_c1) + list(params_p1) + list(params_c2) + list(params_p2),
        estimator=estimator,
    )
    return qnn

class QuantumHybridCNN:
    """
    Quantum side of the hybrid QCNN.  The class exposes a single
    ``build`` method that accepts a NumPy adjacency matrix produced by the
    classical FeatureGraphNet and returns an EstimatorQNN ready for
    training with Qiskit Machine Learning.
    """
    @staticmethod
    def build(adj_matrix: np.ndarray) -> EstimatorQNN:
        return build_qcnn(adj_matrix)

__all__ = ["conv_circuit", "pool_circuit", "build_qcnn", "QuantumHybridCNN"]
