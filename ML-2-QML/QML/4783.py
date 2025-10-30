"""Quantum hybrid regression model using torchquantum and classical preprocessing."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
#  Classical preprocessing utilities (duplicated for consistency)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """PyTorch convolutional filter mimicking a quantum filter."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1, 2, 3])  # (batch,)


# --------------------------------------------------------------------------- #
#  Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum data."""
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
#  Quantum layer
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)


# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(tq.QuantumModule):
    """Quantum hybrid regression combining classical convolution with quantum encoding."""

    def __init__(self, num_wires: int, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.n_wires = num_wires
        self.conv = ConvFilter(kernel_size, threshold)
        self.q_layer = QLayer(num_wires)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        # Compute convolutional scalar feature
        conv_feat = self.conv.run(state_batch).unsqueeze(-1)  # (batch,1)

        # Prepare encoding input: first num_wires-1 dims from state_batch, last from conv_feat
        if state_batch.shape[-1] >= self.n_wires - 1:
            encoder_input = torch.cat([state_batch[..., :self.n_wires - 1], conv_feat], dim=-1)
        else:
            pad = torch.zeros(bsz, self.n_wires - state_batch.shape[-1] - 1, device=state_batch.device)
            encoder_input = torch.cat([state_batch, pad, conv_feat], dim=-1)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, encoder_input)
        self.q_layer(qdev)
        features = self.measure(qdev)  # (batch, n_wires)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
