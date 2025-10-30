from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

from.SelfAttention import SelfAttention

class HybridQCNNGraphAttention:
    """Quantum hybrid QCNN with a self‑attention block.

    The circuit first encodes the input data with a ZFeatureMap,
    then applies a parameterised self‑attention block (adapted from
    the classical SelfAttention helper), followed by the QCNN ansatz
    composed of convolution and pooling layers.  The final expectation
    value of a Z observable is used as the output probability.
    """
    def __init__(self, input_dim: int = 8, shots: int = 1024) -> None:
        self.input_dim = input_dim
        self.shots = shots

        # Attention parameters
        self.attention_params = ParameterVector("θ_att",
                                                length=input_dim * 3 + (input_dim - 1))

        # QCNN quantum part
        from.QCNN import QCNN as QCNN_factory
        self.qcnn_qnn = QCNN_factory()
        self.qcnn_circuit = self.qcnn_qnn.circuit
        self.qcnn_input_params = self.qcnn_qnn.input_params
        self.qcnn_weight_params = self.qcnn_qnn.weight_params

        # Compose full circuit
        attention_circ = self._build_attention_circuit()
        full_circuit = QuantumCircuit(self.input_dim, name="Hybrid QCNN")
        full_circuit.compose(attention_circ, range(self.input_dim), inplace=True)
        full_circuit.compose(self.qcnn_circuit, range(self.input_dim), inplace=True)

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.input_dim - 1), 1)])

        # Estimator and QNN
        estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=full_circuit,
            observables=observable,
            input_params=self.qcnn_input_params,
            weight_params=list(self.attention_params) + list(self.qcnn_weight_params),
            estimator=estimator,
            shots=self.shots,
        )

    def _build_attention_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.input_dim)
        # rotation gates
        for i in range(self.input_dim):
            qc.rx(self.attention_params[3 * i], i)
            qc.ry(self.attention_params[3 * i + 1], i)
            qc.rz(self.attention_params[3 * i + 2], i)
        # entangling gates
        for i in range(self.input_dim - 1):
            qc.crx(self.attention_params[self.input_dim * 3 + i], i, i + 1)
        return qc

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.qnn.predict(inputs.reshape(-1, self.input_dim))

def QCNN() -> HybridQCNNGraphAttention:
    """Factory returning the hybrid quantum QCNN."""
    return HybridQCNNGraphAttention()
