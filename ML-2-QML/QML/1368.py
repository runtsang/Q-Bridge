"""Quantum regression model using torchquantum with a multi‑observable ansatz,
quantum‑kernel helper, and hybrid training loop."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_wires: int,
    samples: int,
    noise_std: float = 0.05,
    shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complex quantum states of the form
    cos(theta)|0..0> + e^{i phi} sin(theta)|1..1> with optional label noise.
    """
    rng = np.random.default_rng()
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = rng.random(samples) * 2 * math.pi
    phis = rng.random(samples) * 2 * math.pi
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis) + shift
    labels += rng.normal(scale=noise_std, size=samples).astype(np.float32)
    return states, labels


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """
    Torch dataset wrapping the complex quantum states.
    """

    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.05, shift: float = 0.0):
        self.states, self.labels = generate_superposition_data(
            num_wires=num_wires,
            samples=samples,
            noise_std=noise_std,
            shift=shift,
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Quantum model
# --------------------------------------------------------------------------- #
class RegressionModel(tq.QuantumModule):
    """
    Quantum‑classical hybrid regression model.
    The quantum part implements a layered entangling ansatz with
    RX, RY, RZ gates, followed by measurement of multiple Pauli observables.
    A classical linear head maps the expectation values to a scalar output.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int = 3):
            super().__init__()
            self.num_wires = num_wires
            self.depth = depth
            self.layers: list[tq.QuantumModule] = []

            for _ in range(depth):
                self.layers.append(tq.RandomLayer(n_ops=20, wires=list(range(num_wires))))
                self.layers.append(tq.RX(has_params=True, trainable=True))
                self.layers.append(tq.RY(has_params=True, trainable=True))
                self.layers.append(tq.RZ(has_params=True, trainable=True))
                self.layers.append(tq.CNOT(wires=[(i, (i + 1) % num_wires) for i in range(num_wires)]))

        def forward(self, qdev: tq.QuantumDevice) -> None:
            for layer in self.layers:
                layer(qdev)

    def __init__(self, num_wires: int, observables: Iterable[str] = ("Z", "X", "Y")):
        super().__init__()
        self.num_wires = num_wires
        self.observables = list(observables)

        # Encoder: amplitude encoding via amplitude‑encoding gate
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])

        # Variational layer
        self.q_layer = self.QLayer(num_wires)

        # Measurement of multiple Pauli observables
        self.measure = tq.MeasureAll(tq.PauliZ)  # placeholder; we will post‑process

        # Classical head
        self.head = nn.Linear(num_wires * len(self.observables), 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)

        # Encode the input state
        self.encoder(qdev, state_batch)

        # Apply variational ansatz
        self.q_layer(qdev)

        # Measure expectation values of selected observables
        exp_vals = []
        for obs in self.observables:
            if obs == "Z":
                meas = tq.MeasureAll(tq.PauliZ)
            elif obs == "X":
                meas = tq.MeasureAll(tq.PauliX)
            elif obs == "Y":
                meas = tq.MeasureAll(tq.PauliY)
            else:
                raise ValueError(f"Unsupported observable: {obs}")
            exp_vals.append(meas(qdev))

        # Concatenate features
        features = torch.cat(exp_vals, dim=-1)

        return self.head(features).squeeze(-1)

    # --------------------------------------------------------------------- #
    # Quantum‑kernel helper
    # --------------------------------------------------------------------- #
    def compute_kernel_matrix(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel matrix K_{ij} = |⟨ψ_i|ψ_j⟩|^2
        using a state‑vector simulator.
        """
        device = states.device
        n_samples = states.shape[0]
        # Allocate a simulator
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=n_samples, device=device)
        self.encoder(qdev, states)
        # Obtain statevectors
        sv = qdev.get_statevector()
        # Compute inner products
        k = torch.abs(torch.einsum("bi,bj->ij", sv, sv.conj())) ** 2
        return k

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopping:
    patience: int = 8
    min_delta: float = 1e-4
    _best_loss: float | None = None
    _patience_counter: int = 0

    def step(self, val_loss: float) -> bool:
        if self._best_loss is None or val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._patience_counter = 0
            return False
        self._patience_counter += 1
        return self._patience_counter >= self.patience


def train_one_epoch(
    model: tq.QuantumModule,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        states = batch["states"].to(device)
        target = batch["target"].to(device)
        optimizer.zero_grad()
        pred = model(states)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * states.size(0)
    return total_loss / len(loader.dataset)


def evaluate(
    model: tq.QuantumModule,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)
            preds.append(model(states).cpu())
            trues.append(target.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    mse = mean_squared_error(trues, preds)
    r2 = r2_score(trues, preds)
    return mse, r2


__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data", "EarlyStopping"]
