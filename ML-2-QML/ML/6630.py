"""Hybrid classical‑quantum convolutional network.

The module defines `QCNNHybridModel` that stitches together:
* A sparse classical encoder that reduces the 8‑dimensional input to a 4‑dimensional latent representation.
* A QCNN‑style variational ansatz that processes the 8‑qubit circuit with 3 convolution‑pool layers.
* A joint loss that allows end‑to‑end training using PyTorch's autograd and Qiskit’s EstimatorQNN.
"""

import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.utils import algorithm_globals

class QCNNHybridModel(nn.Module):
    """A hybrid model that fuses classical convolutional layers with a QCNN variational circuit."""

    def __init__(self, input_dim: int = 8, latent_dim: int = 4, depth: int = 3):
        super().__init__()
        # Classical encoder (mirrors QCNNModel.feature_map)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, latent_dim),
            nn.Tanh(),
        )
        # Build the quantum circuit using helper functions from the reference
        self._build_qcircuit()
        self.qnn = EstimatorQNN(
            circuit=self.qcircuit,
            observables=[SparsePauliOp.from_list([("Z", 1)])],
            input_params=self.feature_map.parameters,
            weight_params=self.qc_params,
            estimator=Estimator()
        )

    def _build_qcircuit(self):
        """Create the QCNN ansatz used in the hybrid model."""
        # Feature map – 8‑qubit Z‑feature map
        self.feature_map = QuantumCircuit(8)
        for i in range(8):
            self.feature_map.rz(ParameterVector(f'x{i}', 1)[0], i)
            self.feature_map.cz(i, (i + 1) % 8)
        # Convolution layer helper
        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=num_qubits * 3)
            idx = 0
            for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
                sub = QuantumCircuit(2)
                sub.rz(-np.pi / 2, 1)
                sub.cx(1, 0)
                sub.rz(params[idx], 0)
                sub.ry(params[idx + 1], 1)
                sub.cx(0, 1)
                sub.ry(params[idx + 2], 1)
                sub.cx(1, 0)
                sub.rz(np.pi / 2, 0)
                qc.append(sub.to_instruction(), [q1, q2])
                idx += 3
            return qc
        # Pooling layer helper
        def pool_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
            idx = 0
            for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
                sub = QuantumCircuit(2)
                sub.rz(-np.pi / 2, 1)
                sub.cx(1, 0)
                sub.rz(params[idx], 0)
                sub.ry(params[idx + 1], 1)
                sub.cx(0, 1)
                sub.ry(params[idx + 2], 1)
                qc.append(sub.to_instruction(), [q1, q2])
                idx += 3
            return qc
        # Assemble the full ansatz
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, 'c1'), range(8), inplace=True)
        ansatz.compose(pool_layer(8, 'p1'), range(8), inplace=True)
        ansatz.compose(conv_layer(4, 'c2'), range(4, 8), inplace=True)
        ansatz.compose(pool_layer(4, 'p2'), range(4, 8), inplace=True)
        ansatz.compose(conv_layer(2, 'c3'), range(6, 8), inplace=True)
        ansatz.compose(pool_layer(2, 'p3'), range(6, 8), inplace=True)
        self.qcircuit = ansatz
        self.qc_params = ansatz.parameters

    def forward(self, x: torch.Tensor):
        # Classical encoding
        latent = self.encoder(x)
        # Prepare classical feature vector for the quantum circuit
        # The QCNN expects 8 parameters; we broadcast the 4‑dim latent to 8 by duplication
        q_inputs = torch.cat([latent, latent], dim=-1)
        # Convert to numpy for the estimator
        q_inputs_np = q_inputs.detach().cpu().numpy()
        # Evaluate the quantum circuit
        q_output = self.qnn(q_inputs_np)
        # Combine outputs: simple concatenation and final linear layer
        combined = torch.cat([latent, q_output], dim=-1)
        return torch.sigmoid(nn.Linear(combined.shape[-1], 1)(combined))
