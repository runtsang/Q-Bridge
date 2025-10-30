"""
Quantum regression model that uses a Pennylane variational circuit
to extract quantum features from the input state.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Utility: generate superposition data
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_wires: int,
    samples: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.
    The target is sin(2 theta) * cos(phi).
    """
    rng = np.random.default_rng(random_state)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary containing the state vector
    under the key ``states`` and the corresponding target under ``target``.
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

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class QuantumRegression__gen140(nn.Module):
    """
    Variational hybrid model that encodes the input state with a
    parameterised quantum circuit and maps the resulting expectation
    values to a scalar output via a classical linear head.
    """
    def __init__(
        self,
        num_wires: int,
        num_layers: int = 2,
        device: str = "default.qubit",
        interface: str = "torch",
    ):
        super().__init__()
        self.num_wires = num_wires
        self.num_layers = num_layers

        # Trainable parameters for the variational layers
        self.params = nn.Parameter(torch.randn(num_layers, num_wires))

        # QNode that returns expectation values of PauliZ on each wire
        self.qnode = qml.QNode(
            self._qcircuit,
            dev=qml.device(device, wires=num_wires),
            interface=interface,
        )

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def _qcircuit(self, inputs: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input state vector of shape ``(num_wires,)``.
        params : torch.Tensor
            Variational parameters of shape ``(num_layers, num_wires)``.
        """
        # Input encoding: rotate each wire according to the input amplitude
        for i in range(self.num_wires):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for l in range(self.num_layers):
            for i in range(self.num_wires):
                qml.RX(params[l, i], wires=i)
                qml.RZ(params[l, i], wires=i)
            # Entangling layer
            for i in range(self.num_wires - 1):
                qml.CNOT(wires=[i, i + 1])

        # Return expectation values of PauliZ on every wire
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Batched input states of shape ``(batch, num_wires)``.
        """
        batch_size = inputs.shape[0]
        # Compute quantum features for each sample in the batch
        q_features = torch.stack(
            [self.qnode(inputs[i], self.params) for i in range(batch_size)]
        )
        # Map to scalar output
        return self.head(q_features).squeeze(-1)

__all__ = ["QuantumRegression__gen140", "RegressionDataset", "generate_superposition_data"]
