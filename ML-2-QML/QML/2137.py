"""Quantum regression model leveraging a variational circuit with hybrid feature extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random qubit superposition states |ψ⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩.
    Labels are derived from the interference term sin(2θ)cosφ.
    """
    dim = 2 ** num_wires
    omega0 = np.zeros(dim, dtype=complex)
    omega0[0] = 1.0
    omega1 = np.zeros(dim, dtype=complex)
    omega1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, dim), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum states and regression targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return torch.tensor(self.states[idx], dtype=torch.cfloat), \
               torch.tensor(self.labels[idx], dtype=torch.float32)

class QuantumRegressionModel(tq.QuantumModule):
    """Variational quantum circuit for regression with hybrid feature extraction."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, n_layers: int = 2):
            super().__init__()
            self.n_wires = num_wires
            self.n_layers = n_layers
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rzz = tq.RZZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.n_layers):
                self.random_layer(qdev)
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)
                # Entangling layer
                for w in range(self.n_wires - 1):
                    self.rzz(qdev, wires=(w, w + 1))

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        # Measure pairwise ZZ correlations
        self.measure_zz = tq.MeasureAll(tq.PauliZ, wires=list(range(num_wires)), correlation=True)
        self.head = nn.Linear(num_wires + num_wires * (num_wires - 1) // 2, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical amplitudes into quantum states
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        # Expectation of Z on each wire
        z_exp = self.measure_z(qdev)
        # Expectation of ZZ correlations
        zz_exp = self.measure_zz(qdev)
        features = torch.cat([z_exp, zz_exp], dim=1)
        return self.head(features).squeeze(-1)

def train_qmodel(model: tq.QuantumModule,
                 dataset: Dataset,
                 epochs: int = 120,
                 lr: float = 1e-3,
                 batch_size: int = 32,
                 device: str | torch.device = "cpu",
                 verbose: bool = True) -> list[float]:
    """
    Training loop for the quantum regression model.
    Returns a list of epoch‑level MSE losses.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for states, targets in loader:
            states, targets = states.to(device), targets.to(device)
            opt.zero_grad()
            preds = model(states)
            loss = criterion(preds, targets)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)
        if verbose and epoch % 20 == 0:
            print(f"[QML] Epoch {epoch:03d}/{epochs:03d} | MSE: {epoch_loss:.4f}")
    return losses

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data", "train_qmodel"]
