"""Quantum version of UnifiedQCNN.

Provides a variational circuit that mirrors the classical architecture
using convolution and pooling layers.  The implementation uses
Qiskit’s EstimatorQNN for efficient state‑vector evaluation.

The class name ``UnifiedQCNN`` matches the classical counterpart so
training code can interchange the backend by calling ``QCNN()``.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# Helper functions for conv and pool layers
def conv_circuit(params):
    """Single 2‑qubit convolution unit."""
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
    """Convolutional layer composed of many 2‑qubit units."""
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        sub = conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        sub = conv_circuit(params[param_index:param_index+3])
        qc.append(sub, [q1, q2])
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params):
    """Single 2‑qubit pooling unit."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(num_qubits, param_prefix):
    """Pooling layer that reduces the qubit count by half."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(0, num_qubits, 2):
        sub = pool_circuit(params[param_index:param_index+3])
        qc.append(sub, [i, i+1])
        qc.barrier()
        param_index += 3
    return qc

# Main UnifiedQCNN quantum class
class UnifiedQCNN:
    """Quantum QCNN variational model.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (input dimension).  Must be a power of two.
    depth : int
        Number of convolution‑pooling stages.
    """
    def __init__(self, num_qubits: int = 8, depth: int = 3):
        self.num_qubits = num_qubits
        self.depth = depth
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(self.num_qubits)
        self.ansatz = self._build_ansatz()
        self.circuit = self._compose_full_circuit()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator
        )

    def _build_ansatz(self):
        """Construct the ansatz with convolution and pooling layers."""
        ansatz = QuantumCircuit(self.num_qubits)
        for stage in range(self.depth):
            ansatz.compose(conv_layer(self.num_qubits, f"c{stage+1}"),
                           inplace=True)
            ansatz.compose(pool_layer(self.num_qubits, f"p{stage+1}"),
                           inplace=True)
        return ansatz

    def _compose_full_circuit(self):
        """Combine feature map and ansatz into a single circuit."""
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)
        return circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Run the quantum circuit on input data.

        Parameters
        ----------
        inputs : np.ndarray of shape (batch, num_qubits)
            Classical feature vectors.

        Returns
        -------
        np.ndarray
            Output predictions (probabilities for binary classification or
            regression values for single‑output regression).
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        preds = self.qnn.predict(inputs)
        return preds

def QCNN() -> UnifiedQCNN:
    """Factory returning a default quantum QCNN."""
    return UnifiedQCNN()

__all__ = ["UnifiedQCNN", "QCNN"]
