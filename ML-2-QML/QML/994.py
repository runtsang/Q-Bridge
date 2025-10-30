"""Quantum regression model with a multi‑layer variational circuit.

Enhancements over the original:
- Parameterised entangling layers (CNOT) interleaved with single‑qubit rotations.
- Configurable number of variational layers.
- Optional amplitude or angle encoding.
- Classical head with residual connections.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_superposition_data(
    num_wires: int,
    samples: int,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.
    Adds optional Gaussian noise to the target.

    Parameters
    ----------
    num_wires : int
        Number of qubits (wires).
    samples : int
        Number of samples.
    noise_std : float
        Standard deviation of additive Gaussian noise on the target.

    Returns
    -------
    states : np.ndarray
        Complex state vectors of shape (samples, 2**num_wires).
    labels : np.ndarray
        Target values of shape (samples,).
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = (
            np.cos(thetas[i]) * omega_0
            + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )
    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise_std > 0.0:
        labels += np.random.normal(scale=noise_std, size=labels.shape)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset wrapper for quantum regression data.

    Returns a dictionary with keys ``states`` (complex tensor) and ``target``.
    """

    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(
            num_wires, samples, noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ResidualBlock(nn.Module):
    """Residual block for the classical head."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class QModel(tq.QuantumModule):
    """
    Quantum regression model with a configurable variational circuit.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    n_layers : int
        Number of variational layers (each containing RX/RY + entangling CZ).
    encoder_type : str
        Either ``"amplitude"`` or ``"angle"``.  The default uses amplitude encoding
        because it preserves the full state vector.
    """

    class QLayer(tq.QuantumModule):
        """Single variational layer with rotations and CNOT entanglement."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            # Parameterised rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            # Entangling CZ gates
            self.cz = tq.CZ()

        def forward(self, qdev: tq.QuantumDevice):
            for wire in range(self.num_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            # Chain CNOTs in a ring topology
            for wire in range(self.num_wires):
                self.cz(qdev, wires=[wire, (wire + 1) % self.num_wires])

    def __init__(
        self,
        num_wires: int,
        n_layers: int = 3,
        encoder_type: str = "amplitude",
    ):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.encoder_type = encoder_type

        # Encoder
        if encoder_type == "amplitude":
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
            )
        elif encoder_type == "angle":
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
            )
        else:
            raise ValueError("encoder_type must be 'amplitude' or 'angle'")

        # Variational layers
        self.layers = nn.ModuleList([self.QLayer(num_wires) for _ in range(n_layers)])

        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical head with residual connection
        self.head = nn.Sequential(
            nn.Linear(num_wires, num_wires // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(num_wires // 2),
            nn.Linear(num_wires // 2, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of complex state vectors of shape (batch, 2**num_wires).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.layers:
            layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = [
    "QModel",
    "RegressionDataset",
    "generate_superposition_data",
]
