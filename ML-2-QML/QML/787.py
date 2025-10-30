"""Hybrid QCNN with classical post‑processing head.

The original QML seed implemented a pure quantum neural network.  Here we
augment it with a lightweight classical neural network that consumes the
expectation values of the quantum circuit and produces the final
classification.  This hybrid approach reduces the number of quantum
parameters while still leveraging quantum feature maps.

The class exposes:
    * `circuit` – the variational quantum circuit.
    * `classical_head` – a small PyTorch network.
    * `forward` – runs the quantum circuit, extracts expectation values
      and feeds them to the classical head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNHybrid:
    """Hybrid QCNN with a classical post‑processing head."""

    def __init__(self, input_dim: int = 8, num_qubits: int = 8) -> None:
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.estimator = Estimator()
        algorithm_globals.random_seed = 12345

        # Build quantum part
        self.feature_map = ZFeatureMap(input_dim)
        self.ansatz = self._build_ansatz(num_qubits)
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

        # Classical head: simple 2‑layer MLP
        self.classical_head = nn.Sequential(
            nn.Linear(num_qubits, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # Optimiser placeholder (to be set externally)
        self.optimizer = None

    def _build_ansatz(self, num_qubits: int) -> QuantumCircuit:
        """Constructs a layered conv‑pool ansatz with shared parameters."""
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

        def pool_circuit(params: ParameterVector) -> QuantumCircuit:
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
            qc = QuantumCircuit(num_qubits)
            param_vec = ParameterVector(prefix, length=num_qubits // 2 * 3)
            for i in range(0, num_qubits, 2):
                qc.append(conv_circuit(param_vec[i // 2 * 3: (i // 2 + 1) * 3]), [i, i + 1])
            return qc

        def pool_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
            qc = QuantumCircuit(num_qubits)
            param_vec = ParameterVector(prefix, length=num_qubits // 2 * 3)
            for i in range(0, num_qubits, 2):
                qc.append(pool_circuit(param_vec[i // 2 * 3: (i // 2 + 1) * 3]), [i, i + 1])
            return qc

        # Assemble the ansatz
        qc = QuantumCircuit(num_qubits)
        qc.compose(self.feature_map, range(num_qubits), inplace=True)

        # Layer 1
        qc.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
        qc.compose(pool_layer(num_qubits, "p1"), range(num_qubits), inplace=True)

        # Layer 2 (half the qubits)
        qc.compose(conv_layer(num_qubits // 2, "c2"), range(num_qubits // 2, num_qubits), inplace=True)
        qc.compose(pool_layer(num_qubits // 2, "p2"), range(num_qubits // 2, num_qubits), inplace=True)

        # Layer 3 (quarter the qubits)
        qc.compose(conv_layer(num_qubits // 4, "c3"), range(num_qubits // 2, num_qubits), inplace=True)
        qc.compose(pool_layer(num_qubits // 4, "p3"), range(num_qubits // 2, num_qubits), inplace=True)

        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantum circuit and pass the expectation values through
        the classical head.
        """
        # Convert torch tensor to numpy
        inputs = x.detach().cpu().numpy()
        # Quantum evaluation
        q_vals = self.qnn.predict(inputs)
        q_tensor = torch.tensor(q_vals, dtype=torch.float32)
        # Classical post‑processing
        return self.classical_head(q_tensor)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def eval(self) -> None:
        """Set both quantum and classical parts to evaluation mode."""
        self.classical_head.eval()

    def train(self) -> None:
        """Set the classical head to training mode."""
        self.classical_head.train()

__all__ = ["QCNNHybrid"]
