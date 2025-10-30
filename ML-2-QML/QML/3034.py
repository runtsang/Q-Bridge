import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QuanvolutionHybrid:
    """Quantum neural network mirroring the classical QuanvolutionHybrid."""
    def __init__(self, patch_size: int = 2, num_patches: int = 196):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_qubits = self.num_patches * self.patch_size
        self.estimator = Estimator()
        self.qnn = self._build_qnn()
    def _build_qnn(self):
        # Feature map: RX rotations with input parameters
        x = ParameterVector('x', length=self.num_qubits)
        feature_map = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            feature_map.rx(x[i], i)
        # Convolutional block
        def conv_circuit(params):
            qc = QuantumCircuit(self.patch_size)
            qc.ry(params[0], 0)
            qc.rz(params[1], 1)
            qc.cx(0, 1)
            return qc
        # Pooling block
        def pool_circuit(params):
            qc = QuantumCircuit(self.patch_size)
            qc.rz(-np.pi/2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc
        # Ansatz: convolution + pooling layers
        ansatz = QuantumCircuit(self.num_qubits)
        # Convolution layer
        for i in range(0, self.num_qubits, 2):
            params = ParameterVector(f'c{i}', length=3)
            conv = conv_circuit(params)
            ansatz.append(conv, [i, i+1])
        # Pooling layer
        for i in range(0, self.num_qubits, 4):
            params = ParameterVector(f'p{i}', length=3)
            pool = pool_circuit(params)
            ansatz.append(pool, [i, i+1, i+2, i+3])
        # Combine feature map and ansatz
        full_circuit = QuantumCircuit(self.num_qubits)
        full_circuit.compose(feature_map, range(self.num_qubits), inplace=True)
        full_circuit.compose(ansatz, range(self.num_qubits), inplace=True)
        # Observable: measure first qubit in Z
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        # Build EstimatorQNN
        qnn = EstimatorQNN(
            circuit=full_circuit,
            observables=observable,
            input_params=x,
            weight_params=ansatz.parameters,
            estimator=self.estimator
        )
        return qnn
    def forward(self, x: torch.Tensor):
        # x shape (batch, 1, 28, 28) → flatten to vector of length num_qubits
        batch = x.shape[0]
        features = x.view(batch, -1).numpy()
        logits = self.qnn.predict(features)
        return torch.tensor(logits, dtype=torch.float32)

__all__ = ["QuanvolutionHybrid"]
