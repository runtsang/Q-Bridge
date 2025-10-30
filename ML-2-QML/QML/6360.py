"""Quantum regression model with Bayesian head and early stopping.

This module extends the original seed by adding a Bayesian linear regression head
that produces predictive mean and log‑variance, enabling uncertainty
estimation.  An EarlyStopping callback is also provided.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random superposition states |ψ⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩.
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

class RegressionDataset(Dataset):
    """
    Dataset that returns quantum states and labels.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict:
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class BayesianLinearHead(nn.Module):
    """
    Bayesian linear regression head for quantum‑derived features.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.randn(in_features, 1) * 0.1)
        self.bias_mu = nn.Parameter(torch.zeros(1))
        self.weight_logvar = nn.Parameter(torch.full((in_features, 1), -5.0))
        self.bias_logvar = nn.Parameter(torch.full((1,), -5.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x @ self.weight_mu + self.bias_mu
        var = x.pow(2) @ torch.exp(self.weight_logvar) + torch.exp(self.bias_logvar)
        logvar = torch.log(var + 1e-8)
        return mean.squeeze(-1), logvar.squeeze(-1)

class QuantumRegressionModel(tq.QuantumModule):
    """
    Quantum regression model with variational circuit and Bayesian head.
    """
    class QLayer(tq.QuantumModule):
        """
        Variational layer with random rotations and trainable RX/RY gates.
        """
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

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = BayesianLinearHead(in_features=num_wires)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing predictive mean and log‑variance.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of quantum state vectors of shape (batch, 2**num_wires).

        Returns
        -------
        mean : torch.Tensor
            Predictive mean.
        logvar : torch.Tensor
            Predictive log‑variance.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # shape (batch, num_wires)
        return self.head(features)

    def loss(self, mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Negative log‑likelihood loss for Gaussian predictions.

        Parameters
        ----------
        mean : torch.Tensor
            Predictive mean.
        logvar : torch.Tensor
            Predictive log‑variance.
        target : torch.Tensor
            Ground‑truth targets.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss.
        """
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + ((target - mean).pow(2) / var))
        return nll.mean()

class EarlyStopping:
    """
    Same early‑stopping utility as in the classical pipeline.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

__all__ = [
    "QuantumRegressionModel",
    "RegressionDataset",
    "generate_superposition_data",
    "BayesianLinearHead",
    "EarlyStopping",
]
