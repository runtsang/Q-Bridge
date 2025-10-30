"""Quantum regression model combining EstimatorQNN and a classical head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset generating quantum states and labels."""
    def __init__(self, samples: int, num_wires: int = 1):
        self.states, self.labels = self._generate_state_data(num_wires, samples)

    def _generate_state_data(self, num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            omega_0 = np.zeros(2 ** num_wires, dtype=complex)
            omega_0[0] = 1.0
            omega_1 = np.zeros(2 ** num_wires, dtype=complex)
            omega_1[-1] = 1.0
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels.astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridQuantumRegression:
    """Hybrid quantum‑classical regression model."""
    def __init__(self, num_wires: int = 1, device: str = "cpu"):
        self.num_wires = num_wires
        # Build a simple EstimatorQNN circuit
        self.input_param = Parameter("theta")
        self.weight_param = Parameter("phi")
        qc = QuantumCircuit(num_wires)
        qc.h(0)
        qc.ry(self.input_param, 0)
        qc.rx(self.weight_param, 0)
        # Observable Y
        observable = SparsePauliOp.from_list([("Y" * num_wires, 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=estimator,
        )
        # Classical head
        self.head = nn.Linear(1, 1)

    def forward(self, theta_batch: torch.Tensor, phi_batch: torch.Tensor) -> torch.Tensor:
        """Run the EstimatorQNN and a classical linear head."""
        # Prepare parameters dict
        param_dict = {self.input_param: theta_batch.numpy(), self.weight_param: phi_batch.numpy()}
        # Evaluate
        expectation = self.estimator_qnn.evaluate(param_dict)
        # Convert to torch tensor
        exp_tensor = torch.tensor(expectation, dtype=torch.float32, device=theta_batch.device)
        return self.head(exp_tensor.unsqueeze(-1)).squeeze(-1)

__all__ = ["HybridQuantumRegression", "RegressionDataset"]
