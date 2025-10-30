"""QuantumRegression__gen082.py – quantum regression with hybrid loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]


def generate_superposition_data(
    num_wires: int,
    samples: int,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics a single‑qubit superposition
    encoded across *num_wires* qubits.  The state for each sample is

        |ψ⟩ = cos(θ)|0…0⟩ + e^{iϕ} sin(θ)|1…1⟩,

    where θ and ϕ are drawn uniformly from [0, 2π).  The target label is
    y = sin(2θ) * cos(ϕ).  Optional Gaussian noise can be added to the label.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise_std > 0.0:
        labels += np.random.normal(scale=noise_std, size=labels.shape).astype(np.float32)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset for quantum regression.  Each item is a dict containing:

    - ``states``: a complex‑valued state vector as a torch tensor (cfloat)
    - ``target``: regression target as a torch tensor (float32)
    """

    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(
            num_wires=num_wires, samples=samples, noise_std=noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """
    Hybrid quantum‑classical regression model.

    The architecture consists of:

    1. **Encoder** – a *GeneralEncoder* that maps the raw state onto a
       computational basis representation suitable for measurement.
    2. **Variational layer** – a random layer followed by a trainable RX/RY pair
       on each wire.  This implements a universal circuit block.
    3. **Measurement** – Pauli‑Z expectation values on all wires.
    4. **Head** – a classical linear layer that maps the measurement vector
       to a scalar prediction.
    5. **Hybrid loss** – the forward method returns both the quantum prediction
       and an auxiliary probability that the target lies in the upper quartile.
       The loss can be a weighted sum of the MSE and a binary cross‑entropy
       term, enabling regularisation of the quantum feature extractor.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, dropout: float = 0.1):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning:

        - ``pred``: quantum regression prediction.
        - ``aux``: auxiliary probability that target is in upper quartile.

        The auxiliary probability is computed from the same feature vector
        used for the primary head, providing regularisation without
        requiring extra parameters.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        features = self.dropout(features)
        pred = self.head(features).squeeze(-1)
        aux = torch.sigmoid(self.head(features)).squeeze(-1)  # reuse head weights
        return pred, aux
