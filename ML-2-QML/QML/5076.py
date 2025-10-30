"""QCNNHybridModel – a quantum implementation of the hybrid QCNN architecture."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.library import ZFeatureMap
from qiskit import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper circuits
# --------------------------------------------------------------------------- #
def conv_circuit(params):
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

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = conv_circuit(params[i*3:(i+1)*3])
        qc.append(sub, [i, i+1])
    return qc

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
    for idx, (src, snk) in enumerate(zip(sources, sinks)):
        sub = pool_circuit(params[idx*3:(idx+1)*3])
        qc.append(sub, [src, snk])
    return qc

# --------------------------------------------------------------------------- #
# QCNN hybrid quantum model
# --------------------------------------------------------------------------- #
class QCNNHybridModel(EstimatorQNN):
    """Quantum version of the hybrid QCNN architecture using EstimatorQNN."""
    def __init__(self, num_qubits: int = 8, seed: int | None = None):
        if seed is not None:
            algorithm_globals.random_seed = seed

        # Feature map
        feature_map = ZFeatureMap(num_qubits)

        # Ansatz construction
        ansatz = QuantumCircuit(num_qubits)

        # 1st convolution + pooling
        ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
        ansatz.compose(pool_layer(list(range(num_qubits // 2)), list(range(num_qubits // 2, num_qubits)), "p1"), inplace=True)

        # 2nd convolution + pooling
        ansatz.compose(conv_layer(num_qubits // 2, "c2"), inplace=True)
        ansatz.compose(pool_layer(list(range(num_qubits // 4)), list(range(num_qubits // 4, num_qubits // 2)), "p2"), inplace=True)

        # 3rd convolution + pooling
        ansatz.compose(conv_layer(num_qubits // 4, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        estimator = StatevectorEstimator()

        super().__init__(
            circuit=circuit,
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

# Factory for backward compatibility
def QCNNHybridModelFactory() -> QCNNHybridModel:
    """Return a freshly constructed QCNNHybridModel quantum instance."""
    return QCNNHybridModel()

__all__ = [
    "QCNNHybridModel",
    "QCNNHybridModelFactory",
]
