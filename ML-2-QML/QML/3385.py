"""Quantum estimator that fuses the QCNN ansatz with a lightweight
input encoding inspired by EstimatorQNN.  The circuit consists of
an 8‑qubit feature map followed by three convolution‑pool layers,
and a single‑qubit rotation that serves as the trainable weight.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNN:
    """Wrapper around Qiskit EstimatorQNN that combines a QCNN‑style ansatz
    with a single‑qubit rotation weight.
    """

    def __init__(self):
        # 1‑qubit weight rotation
        weight = Parameter("θ")
        weight_circuit = QuantumCircuit(1)
        weight_circuit.rx(weight, 0)

        # Feature map for 8‑qubit input
        feature_map = ZFeatureMap(8, reps=1, entanglement="linear")

        # Convolution‑pool ansatz
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

        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits * 3)
            idx = 0
            for i in range(0, num_qubits, 2):
                sub = conv_circuit(params[idx:idx+3])
                qc.append(sub.to_instruction(), [i, i+1])
                idx += 3
            return qc

        def pool_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits * 3)
            idx = 0
            for i in range(0, num_qubits, 2):
                sub = pool_circuit(params[idx:idx+3])
                qc.append(sub.to_instruction(), [i, i+1])
                idx += 3
            return qc

        # Build ansatz
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, "c1"), inplace=True)
        ansatz.compose(pool_layer(8, "p1"), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), inplace=True)
        ansatz.compose(pool_layer(4, "p2"), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), inplace=True)
        ansatz.compose(pool_layer(2, "p3"), inplace=True)

        # Combine feature map, ansatz and weight rotation
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        circuit.compose(weight_circuit, inplace=True)

        # Observable: Z on first qubit
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        estimator = StatevectorEstimator()
        self.qnn = QiskitEstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=[weight],
            estimator=estimator,
        )

    def predict(self, X):
        """Delegate prediction to the underlying Qiskit EstimatorQNN."""
        return self.qnn.predict(X)

__all__ = ["EstimatorQNN"]
