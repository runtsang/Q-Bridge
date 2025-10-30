"""
QuantumFraudDetectionCircuit: QCNN‑style variational circuit for fraud detection.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# ------------------------------------------------------------------
#  Construction helpers
# ------------------------------------------------------------------
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

def _layer(num_qubits: int, layer_fn, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    idx = 0
    for i in range(0, num_qubits, 2):
        sub = layer_fn(params[idx:idx+3])
        qc.append(sub, [i, i+1])
        qc.barrier()
        idx += 3
    return qc

# ------------------------------------------------------------------
#  QCNN variational circuit
# ------------------------------------------------------------------
def build_quantum_qcnn(num_qubits: int = 8) -> EstimatorQNN:
    """
    Builds a QCNN variational circuit mirroring the classical QCNN
    structure: convolution → pooling → convolution → pooling → convolution.
    """
    feature_map = qiskit.circuit.library.ZFeatureMap(num_qubits)
    circuit = QuantumCircuit(num_qubits)

    # First convolution
    circuit.append(_layer(num_qubits, _conv_circuit, "c1"), range(num_qubits))
    # First pooling
    circuit.append(_layer(num_qubits, _pool_circuit, "p1"), range(num_qubits))

    # Second convolution on remaining 4 qubits
    circuit.append(_layer(num_qubits//2, _conv_circuit, "c2"), list(range(num_qubits//2, num_qubits)))
    # Second pooling
    circuit.append(_layer(num_qubits//2, _pool_circuit, "p2"), list(range(num_qubits//2, num_qubits)))

    # Third convolution on 2 qubits
    circuit.append(_layer(num_qubits//4, _conv_circuit, "c3"), list(range(num_qubits//2, num_qubits)))

    # Third pooling
    circuit.append(_layer(num_qubits//4, _pool_circuit, "p3"), list(range(num_qubits//2, num_qubits)))

    # Combine feature map and ansatz
    circuit.compose(feature_map, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I"*(num_qubits-1), 1)])
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

# ------------------------------------------------------------------
#  Wrapper for inference
# ------------------------------------------------------------------
class QuantumFraudDetection:
    """
    Encapsulates the QCNN EstimatorQNN and provides a `predict`
    method compatible with the classical hybrid forward pass.
    """
    def __init__(self, num_qubits: int = 8):
        self.qnn = build_quantum_qcnn(num_qubits)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, features) where features == num_qubits
        Returns probabilities in [0,1].
        """
        probs = self.qnn.predict(x)
        return probs.reshape(-1, 1)

__all__ = ["QuantumFraudDetection"]
