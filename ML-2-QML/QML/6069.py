"""UnifiedQCNN – Quantum implementation using Qiskit.

The quantum side implements a variational QCNN that looks at a
convolution‑pooling architecture.  The circuit is built from the
same parameter ordering used in the classical model so that
parameter‑wise comparison is easy.  The ansatz is a
state‑vector estimator (an EstimatorQNN) that can be trained
with classical optimizers (COBYLA, L-BFGS, etc.).  The
graph‑based fidelity analysis is reused from GraphQNN.py.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.circuit.library import ZFeatureMap

# --------------------------------------------------------------------------- #
# 1.  Convolution and pooling primitives
# --------------------------------------------------------------------------- #
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unitary used in the QCNN."""
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

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    """Apply a convolution unitary to each adjacent pair of qubits."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [i, i+1])
        param_index += 3
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unitary – identical to the convolution except
    the last RZ gate is omitted."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    """Pool two source qubits into a single sink qubit."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        sub = _pool_circuit(params[param_index:param_index+3])
        qc.append(sub, [src, snk])
        param_index += 3
    return qc

# --------------------------------------------------------------------------- #
# 2.  Full QCNN ansatz
# --------------------------------------------------------------------------- #
def _qc_cnn_ansatz() -> QuantumCircuit:
    """Build the full 8‑qubit QCNN ansatz."""
    qc = QuantumCircuit(8)

    # Feature map
    feature_map = ZFeatureMap(8)
    qc.append(feature_map, range(8))

    # First convolution and pooling
    qc.append(conv_layer(8, "c1"), range(8))
    qc.append(pool_layer([0,1,2,3], [4,5,6,7], "p1"), range(8))

    # Second convolution and pooling
    qc.append(conv_layer(4, "c2"), [4,5,6,7])
    qc.append(pool_layer([0,1], [2,3], "p2"), [4,5,6,7])

    # Third convolution and pooling
    qc.append(conv_layer(2, "c3"), [6,7])
    qc.append(pool_layer([0], [1], "p3"), [6,7])

    return qc.decompose()

# --------------------------------------------------------------------------- #
# 3.  EstimatorQNN wrapper
# --------------------------------------------------------------------------- #
def UnifiedQCNN() -> EstimatorQNN:
    """
    Return a Qiskit EstimatorQNN that implements the QCNN architecture.
    The parameter ordering of the ansatz matches the classical
    UnifiedQCNNModel, facilitating direct comparison of gradients.
    """
    estimator = StatevectorEstimator()
    circuit = _qc_cnn_ansatz()

    # Observable for binary classification
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Build the EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=ZFeatureMap(8).parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

# --------------------------------------------------------------------------- #
# 4.  Graph diagnostics (re‑exported from GraphQNN)
# --------------------------------------------------------------------------- #
from GraphQNN import (
    feedforward as _feedforward,
    fidelity_adjacency as _fidelity_adjacency,
    random_network as _random_network,
    random_training_data as _random_training_data,
    state_fidelity as _state_fidelity,
)

__all__ = ["UnifiedQCNN", "conv_layer", "pool_layer", "_conv_circuit",
           "_pool_circuit", "_qc_cnn_ansatz", "_feedforward",
           "_fidelity_adjacency", "_random_network",
           "_random_training_data", "_state_fidelity"]
