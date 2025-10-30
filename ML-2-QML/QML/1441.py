"""Quantum regression model built with PennyLane and amplitude‑encoding."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate amplitude‑encoded states |ψ⟩ = cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩.
    Labels are a smooth non‑linear function of θ and ϕ.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i, 0] = np.cos(thetas[i])
        states[i, -1] = np.exp(1j * phis[i]) * np.sin(thetas[i])
    labels = np.sin(2 * thetas) * np.cos(phis)
    # Add a small amount of noise to emulate imperfect state preparation
    labels += 0.05 * np.random.randn(samples).astype(np.float32)
    return states.astype(complex), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset providing amplitude‑encoded states and target values.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QRegressionModel(nn.Module):
    """
    Quantum neural network implemented with PennyLane.
    The network uses amplitude encoding, a variational layer with
    entanglement, and a linear head to map expectation values to a scalar.
    """
    def __init__(self, num_wires: int, device: str = "default.qubit"):
        super().__init__()
        self.num_wires = num_wires
        self.dev = qml.device(device, wires=num_wires, shots=1024)

        # Register trainable parameters for the variational layer
        self.params = nn.Parameter(pnp.random.randn(1, num_wires, 3))

        # Linear head to collapse expectation values to output
        self.head = nn.Linear(num_wires, 1)

        # Create the QNode once; the parameters are passed at call time
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(in_state: torch.Tensor, var_params: torch.Tensor) -> torch.Tensor:
            # Amplitude encoding
            qml.StatePrep(in_state, wires=range(num_wires))
            # Variational entangling block
            for i in range(num_wires):
                qml.RX(var_params[0, i, 0], wires=i)
                qml.RY(var_params[0, i, 1], wires=i)
                qml.RZ(var_params[0, i, 2], wires=i)
            for i in range(num_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of PauliZ on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of amplitude‑encoded states through the variational circuit
        and maps the resulting expectation values to a scalar output.
        """
        # Ensure batch dimension is handled
        expvals = self.circuit(state_batch, self.params)
        expvals = torch.stack(expvals, dim=1)  # shape: (batch, wires)
        return self.head(expvals).squeeze(-1)


__all__ = ["QRegressionModel", "RegressionDataset", "generate_superposition_data"]
