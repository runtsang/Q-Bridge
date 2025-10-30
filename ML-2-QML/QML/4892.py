"""Quantum regression model combining a variational circuit and a quantum kernel."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data generation & dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states |ψ(θ,ϕ)⟩ = cosθ|0…0⟩ + e^{iϕ}sinθ|1…1⟩."""
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
    """Dataset exposing quantum states and regression targets."""
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
# Quantum building blocks
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """Fixed ansatz used as a feature encoder for the kernel."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = tq.RandomLayer(
            n_ops=30,
            wires=list(range(n_wires)),
            seed=42,
            has_params=False,
        )

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        qdev.reset_states(x.shape[0])
        self.ansatz(qdev)
        # Placeholder: a proper quantum kernel would use state overlap

# --------------------------------------------------------------------------- #
# Variational regression circuit
# --------------------------------------------------------------------------- #
class QRegressionCircuit(tq.QuantumModule):
    """Variational ansatz with data encoding and trainable layers."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{n_wires}xRy"]
        )
        self.var_layer = tq.RandomLayer(
            n_ops=25,
            wires=list(range(n_wires)),
            has_params=True,
            trainable=True,
            seed=7,
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, state_batch: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, state_batch)
        self.var_layer(qdev)
        return self.measure(qdev)

class QuantumRegressionModel(tq.QuantumModule):
    """
    Quantum regression model that emulates the classical pipeline:
        1. Data encoding via GeneralEncoder
        2. Variational feature extraction
        3. Quantum kernel evaluation (placeholder)
        4. Classical linear head
    """
    def __init__(self, n_wires: int, head_dim: int = 1):
        super().__init__()
        self.n_wires = n_wires
        self.circuit = QRegressionCircuit(n_wires)
        self.kernel = QuantumKernel(n_wires)
        self.head = nn.Linear(n_wires, head_dim)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)

        # 1. Process states through the variational circuit
        features = self.circuit(qdev, state_batch)

        # 2. Apply a simple quantum kernel by re‑encoding the same batch
        #    (here we use the raw features as a proxy)
        kernel_features = features.detach().float()

        # 3. Linear regression head
        return self.head(kernel_features).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
