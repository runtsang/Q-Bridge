"""Hybrid quantum QCNN that mirrors the classical architecture using Qiskit circuits.

The quantum model implements convolution and pooling layers as parameterized two‑qubit
unitaries, a feature map, and an EstimatorQNN wrapper for easy evaluation.  It shares
the same public interface as the classical version so that experiments can be swapped
between the two back‑ends.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# Helper functions to build convolution and pooling layers
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[i*3:(i+1)*3]), [i, i+1])
    return qc

def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk, p in zip(sources, sinks, params):
        sub = pool_circuit(p)
        qc.append(sub, [src, snk])
    return qc

# Quantum hybrid model
class HybridQCNN:
    def __init__(self) -> None:
        algorithm_globals.random_seed = 12345
        self.estimator = StatevectorEstimator()
        self._build_circuit()

    def _build_circuit(self) -> None:
        # Feature map
        self.feature_map = ZFeatureMap(8)
        # Ansatz construction
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)
        # Combine
        circuit = QuantumCircuit(8)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        self.circuit = circuit.decompose()
        # Observable
        self.obs = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
        # EstimatorQNN wrapper
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.obs,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return expectation values for a batch of classical inputs."""
        return self.qnn.predict(inputs)

def QCNN() -> HybridQCNN:
    """Factory returning the configured hybrid quantum QCNN."""
    return HybridQCNN()

__all__ = ["HybridQCNN", "QCNN"]
