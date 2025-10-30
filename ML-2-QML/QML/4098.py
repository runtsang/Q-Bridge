"""HybridFCLModel – Quantum implementation."""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml
from torch import nn
from torch.utils.data import Dataset
from typing import Tuple


class HybridFCLQuantum(nn.Module):
    """
    Quantum neural network that supports both classification and regression.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 1,
        task: str = "classification",
        use_random: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        num_qubits : int
            Number of qubits (input dimension).
        depth : int, optional
            Number of variational layers. Defaults to 1.
        task : str, optional
            Either ``"classification"`` or ``"regression"``. Defaults to ``"classification"``.
        use_random : bool, optional
            If True, append a random rotation layer after the variational block.
        """
        super().__init__()
        self.task = task
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_random = use_random

        # Device
        self.dev = qml.device("default.qubit", wires=num_qubits)

        # Parameters for variational layers
        self.params = nn.Parameter(
            torch.randn(depth * num_qubits, num_qubits, dtype=torch.float32)
        )

        # Build qnode
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encoding: RX gates
            for i in range(num_qubits):
                qml.RX(inputs[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(depth):
                for i in range(num_qubits):
                    qml.RY(params[idx, i], wires=i)
                    idx += 1
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Optional random rotation layer
            if self.use_random:
                for i in range(num_qubits):
                    qml.RX(0.1 * torch.rand(1), wires=i)

            # Measurements: expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

        # Classical head
        self.head = nn.Linear(num_qubits, 2 if task == "classification" else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, num_qubits)``.
        Returns
        -------
        torch.Tensor
            Output logits (log‑softmax for classification) or scalar predictions.
        """
        # Compute circuit outputs for each sample
        batch_features = torch.stack([self.circuit(sample, self.params) for sample in x])
        out = self.head(batch_features)
        if self.task == "classification":
            out = torch.log_softmax(out, dim=-1)
        return out

    @staticmethod
    def generate_superposition_data(num_qubits: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same data generator as in the quantum regression seed.
        Returns states as complex amplitudes and labels.
        """
        omega_0 = np.zeros(2 ** num_qubits, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_qubits, dtype=complex)
        omega_1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_qubits), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels.astype(np.float32)


class QuantumRegressionDataset(Dataset):
    """
    Dataset that wraps the quantum superposition states for regression.
    """

    def __init__(self, samples: int, num_qubits: int) -> None:
        self.states, self.labels = HybridFCLQuantum.generate_superposition_data(
            num_qubits, samples
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


__all__ = ["HybridFCLQuantum", "QuantumRegressionDataset"]
