import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNHybridModel(nn.Module):
    """
    Quantum circuit-based model that implements a QCNN with convolution
    and pooling layers. The model is wrapped in a PyTorch nn.Module
    so it can be used seamlessly in a hybrid training loop.
    """
    def __init__(self, qubits: int = 8):
        super().__init__()
        self.qubits = qubits
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(qubits)
        self.circuit = self._build_ansatz(qubits)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        self.sigmoid = nn.Sigmoid()

    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """
        Build a QCNN ansatz consisting of alternating convolution
        layers, adapted from the original QCNN example.
        """
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
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for i in range(0, num_qubits, 2):
                sub = conv_circuit(params[(i // 2) * 3 : ((i // 2) + 1) * 3])
                qc.compose(sub, [i, i + 1], inplace=True)
                qc.barrier()
            return qc

        qc = QuantumCircuit(num_qubits)
        qc.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that converts a batch of input data into
        expectation values from the quantum circuit.
        """
        outputs = self.qnn(inputs)
        return self.sigmoid(outputs)
