import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN

class ConvFilter(nn.Module):
    """Drop‑in classical approximation of a quanvolution filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class HybridQCNN(nn.Module):
    """Hybrid classical‑quantum convolutional network."""
    def __init__(self, num_qubits: int = 8, depth: int = 3) -> None:
        super().__init__()
        # Classical feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, num_qubits),
            nn.Tanh()
        )
        # Quantum layer
        self.feature_map = ZFeatureMap(num_qubits)
        self.ansatz = self._build_ansatz(num_qubits, depth)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self, num_qubits: int, depth: int) -> QuantumCircuit:
        """Constructs a convolution‑plus‑pooling ansatz."""
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
            params = ParameterVector(param_prefix, length=num_qubits * 3 // 2)
            idx = 0
            for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
                sub = conv_circuit(params[idx:idx + 3])
                qc.append(sub, [q1, q2])
                idx += 3
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

        def pool_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            idx = 0
            for q1, q2 in zip(range(num_qubits // 2), range(num_qubits // 2, num_qubits)):
                sub = pool_circuit(params[idx:idx + 3])
                qc.append(sub, [q1, q2])
                idx += 3
            return qc

        ansatz = QuantumCircuit(num_qubits)
        ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
        ansatz.compose(pool_layer(num_qubits, "p1"), inplace=True)
        for d in range(2, depth + 1):
            ansatz.compose(conv_layer(num_qubits, f"c{d}"), inplace=True)
            ansatz.compose(pool_layer(num_qubits, f"p{d}"), inplace=True)
        return ansatz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        # Evaluate quantum layer (use zero initial weights for inference)
        weight_vals = {p: 0.0 for p in self.ansatz.parameters()}
        q_expect = self.qnn.evaluate(inputs=features, parameters=weight_vals)
        return torch.sigmoid(q_expect)
