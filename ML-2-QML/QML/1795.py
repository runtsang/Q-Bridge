\
"""Enhanced quantum regression model with entanglement, parameterised layers and optional depolarising noise."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Sequence, Optional

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct states of the form
        cos(theta)|0...0⟩ + e^{i phi} sin(theta)|1...1⟩
    and corresponding labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of training examples.

    Returns
    -------
    states : np.ndarray
        Shape (samples, 2**num_wires), complex128.
    labels : np.ndarray
        Shape (samples,), float32.
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
    return states, labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(Dataset):
    """
    Dataset yielding complex state vectors and real labels.
    """
    def __init__(self, samples: int, num_wires: int, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples)
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = torch.tensor(self.states[idx], dtype=torch.cfloat)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return {"states": x, "target": y}

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(tq.QuantumModule):
    """
    Variational quantum circuit with entanglement, random layers and a classical head.
    Supports optional depolarising noise on measurement outcomes.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Simple entangling chain (CNOTs)
            self.entangle = tq.Sequential(
                *[tq.CNOT(wires=[i, i + 1]) for i in range(n_wires - 1)]
            )
            # Randomised gates to break symmetry
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            # Parameterised single‑qubit rotations (trainable)
            self.param_circuit = tq.Sequential(
                *[tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )

        def forward(self, qdev: tq.QuantumDevice) -> tq.QuantumDevice:
            self.entangle(qdev)
            self.random_layer(qdev)
            self.param_circuit(qdev)
            return qdev

    def __init__(self, num_wires: int, noise_prob: float = 0.0):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.noise_prob = noise_prob

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Execute the circuit for a batch of states.

        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape (batch, 2**n_wires), complex64.

        Returns
        -------
        torch.Tensor
            Predictions of shape (batch,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        if self.noise_prob > 0.0:
            # Simple depolarising noise on measurement outcomes
            noise = torch.randn_like(features, device=features.device) * self.noise_prob
            features = features + noise
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_model(
    model: tq.QuantumModule,
    dataset: Dataset,
    *,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 1e-3,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> tq.QuantumModule:
    """
    Train the variational circuit using MSE loss and Adam optimiser.
    """
    device = torch.device(device or "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        scheduler.step()
        if verbose:
            print(f"[{epoch}/{epochs}] loss: {epoch_loss/len(dataset):.4f}")
    return model

def evaluate_model(
    model: tq.QuantumModule,
    dataset: Dataset,
    *,
    batch_size: int = 32,
    device: torch.device | str | None = None,
) -> float:
    """
    Return the mean‑squared error on ``dataset``.
    """
    device = torch.device(device or "cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        total, count = 0.0, 0
        for batch in loader:
            states = batch["states"].to(device)
            targets = batch["target"].to(device)
            preds = model(states)
            total += criterion(preds, targets).item() * states.size(0)
            count += states.size(0)
    return total / count

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumRegressionModel",
    "train_model",
    "evaluate_model",
]
