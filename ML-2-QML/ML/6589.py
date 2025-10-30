import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNHybridModel(nn.Module):
    """
    Hybrid classical‑quantum convolutional network.
    Architecture:
        1. Classical encoder (Linear → ReLU)
        2. Quantum convolutional block (variational ansatz)
        3. Classical pooling (Linear → ReLU)
        4. Final linear layer for prediction
    The quantum block uses Qiskit's EstimatorQNN and can be executed on a simulator or real backend.
    """

    def __init__(self,
                 n_features: int = 8,
                 n_qubits: int = 8,
                 n_layers: int = 3,
                 n_classes: int = 1):
        super().__init__()
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU()
        )

        # Classical pooling
        self.pool = nn.Sequential(
            nn.Linear(16, 12),
            nn.ReLU()
        )

        # Build the quantum circuit and EstimatorQNN
        self.qnn = self._build_qnn()

        # Final classifier
        self.classifier = nn.Linear(12 + 1, n_classes)  # +1 for quantum expectation

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="ConvLayer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i*3:(i+2)*3])
            qc.append(sub, [i, i+1])
            qc.barrier()
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="PoolLayer")
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for i in range(0, num_qubits, 2):
            sub = self._pool_circuit(params[(i//2)*3:(i//2)*3+3])
            qc.append(sub, [i, i+1])
            qc.barrier()
        return qc

    def _build_qnn(self) -> EstimatorQNN:
        # Feature map
        feature_map = QuantumCircuit(self.n_features)
        for i in range(self.n_features):
            feature_map.h(i)
            feature_map.rz(ParameterVector(f"φ{i}")[0], i)

        # Ansatz
        ansatz = QuantumCircuit(self.n_qubits)
        for layer in range(self.n_layers):
            ansatz.append(self._conv_layer(self.n_qubits, f"c{layer}"), range(self.n_qubits))
            ansatz.append(self._pool_layer(self.n_qubits, f"p{layer}"), range(self.n_qubits))

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.n_features + self.n_qubits)
        circuit.compose(feature_map, range(self.n_features), inplace=True)
        circuit.compose(ansatz, range(self.n_features, self.n_features + self.n_qubits), inplace=True)

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])

        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator
        )
        return qnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical branch
        x_enc = self.encoder(x)
        x_pool = self.pool(x_enc)

        # Quantum branch
        # Convert to numpy for the EstimatorQNN
        x_np = x.detach().cpu().numpy()
        q_expect = self.qnn.predict(x_np)[0]  # expectation value
        q_tensor = torch.tensor(q_expect, dtype=x.dtype, device=x.device)

        # Concatenate and classify
        out = torch.cat([x_pool, q_tensor.unsqueeze(-1)], dim=-1)
        return self.classifier(out)

def QCNN() -> QCNNHybridModel:
    """Factory returning a fully configured instance of QCNNHybridModel."""
    return QCNNHybridModel()
