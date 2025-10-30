"""Quantum regression model combining quantum autoencoder, transformer, and kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq

# Data generation (same as anchor)
def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic regression data in the computational basis."""
    omega_0 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega_0[0] = 1.0
    omega_1 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega_1[-1] = 1.0
    thetas = 2 * torch.pi * torch.rand(samples)
    phis = 2 * torch.pi * torch.rand(samples)
    states = torch.zeros((samples, 2 ** num_wires), dtype=torch.cfloat)
    for i in range(samples):
        states[i] = torch.cos(thetas[i]) * omega_0 + torch.exp(1j * phis[i]) * torch.sin(thetas[i]) * omega_1
    labels = torch.sin(2 * thetas) * torch.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum state vectors and scalar targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {"states": self.states[idx], "target": self.labels[idx]}

# Quantum autoencoder
class QuantumAutoencoder(tq.QuantumModule):
    """Encode classical data into a lower‑dimensional quantum state."""
    def __init__(self, num_qubits: int, latent_dim: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        # Encode input as rotations on each qubit
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(num_qubits)]
        )
        # Trainable unitary (simple RX‑like circuit)
        self.trainable = tq.QuantumLayer(n_ops=2 * num_qubits,
                                         n_wires=num_qubits,
                                         has_params=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, x)
        self.trainable(qdev)
        return self.measure(qdev)

# Quantum transformer block (simplified)
class QuantumTransformerBlock(tq.QuantumModule):
    """Quantum block that applies a trainable circuit to the latent representation."""
    def __init__(self, embed_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.trainable = tq.QuantumLayer(n_ops=2 * n_qubits,
                                         n_wires=n_qubits,
                                         has_params=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.dropout = nn.Dropout(dropout)

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, x)
        self.trainable(qdev)
        return self.measure(qdev)

# Quantum kernel
class QuantumKernel(tq.QuantumModule):
    """Fixed quantum kernel based on a simple Ry ansatz."""
    def __init__(self, num_qubits: int = 4):
        super().__init__()
        self.num_qubits = num_qubits
        self.q_device = tq.QuantumDevice(n_wires=num_qubits)
        self.ansatz = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(num_qubits)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ansatz(self.q_device, x)
        self.ansatz(self.q_device, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# Main quantum regression model
class QuantumRegressionModel(tq.QuantumModule):
    """Hybrid quantum regression model with autoencoder, transformer, and linear head."""
    def __init__(self, num_qubits: int, latent_dim: int = 4,
                 transformer_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        self.autoencoder = QuantumAutoencoder(num_qubits, latent_dim)
        self.transformer = nn.Sequential(*[
            QuantumTransformerBlock(latent_dim, num_qubits, dropout)
            for _ in range(transformer_blocks)
        ])
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.autoencoder.num_qubits,
                                bsz=bsz,
                                device=state_batch.device)
        latent = self.autoencoder(qdev, state_batch)
        latent = latent.unsqueeze(1)  # sequence length 1
        for block in self.transformer:
            latent = block(qdev, latent)
        latent = latent.mean(dim=1)
        return self.regressor(latent).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset",
           "generate_superposition_data", "QuantumKernel"]
