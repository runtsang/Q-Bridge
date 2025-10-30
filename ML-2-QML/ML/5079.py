"""Hybrid classical‑quantum model integrating convolution, quantum kernel, and regression utilities.

This module defines a PyTorch model that mirrors the original quanvolution example
but augments it with a quantum kernel acting on flattened patches and a regression
head.  It also bundles dataset generation and a lightweight estimator for quick
benchmarking.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torch.utils.data import Dataset

__all__ = [
    "QuanvolutionHybrid",
    "QuantumFilter",
    "FastBaseEstimator",
    "generate_superposition_data",
    "RegressionDataset",
    "RegressionModel",
]


# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic data in the form |ψ(θ,φ)⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩.
    The target is a noisy trigonometric function of the angles.
    """
    thetas = np.random.uniform(0, 2 * np.pi, size=(samples,))
    phis = np.random.uniform(0, 2 * np.pi, size=(samples,))
    states = np.zeros((samples, 2 ** num_features), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * np.eye(1, 2 ** num_features, 0).flatten() + \
                    np.exp(1j * phis[i]) * np.sin(thetas[i]) * np.eye(1, 2 ** num_features, -1).flatten()
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Simple dataset for regression that feeds complex state vectors."""

    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """A simple feed‑forward regressor that can be trained on the synthetic data."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Fast estimator utilities
# --------------------------------------------------------------------------- #
class _ensure_batch:
    """Internal helper to turn a list of parameters into a tensor."""

    @staticmethod
    def batch(values: list[float]) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor


class FastBaseEstimator:
    """Evaluate a neural network for batches of parameters and observables."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: list[callable | torch.Tensor],
        parameter_sets: list[list[float]],
    ) -> list[list[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: list[list[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch.batch(params)
                outputs = self.model(inputs)
                row: list[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: list[callable | torch.Tensor],
        parameter_sets: list[list[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: list[list[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Quantum filter used inside the hybrid model
# --------------------------------------------------------------------------- #
class QuantumFilter(nn.Module):
    """Apply a two‑qubit quantum kernel to each flattened patch."""

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect shape (batch, n_qubits) where each row corresponds to a patch.
        batch, _ = x.shape
        qdev = tq.QuantumDevice(self.n_qubits, bsz=batch, device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
# Hybrid classical‑quantum model
# --------------------------------------------------------------------------- #
class QuanvolutionHybrid(nn.Module):
    """Classical convolution + quantum kernel + fully‑connected head for image classification."""

    def __init__(self, in_channels: int = 1, n_qubits: int = 4, n_classes: int = 10):
        super().__init__()
        self.classical_conv = nn.Conv2d(in_channels, n_qubits, kernel_size=2, stride=2)
        self.quantum_filter = QuantumFilter(n_qubits=n_qubits)
        self.fc = nn.Linear(n_qubits * 14 * 14, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical convolution
        x = self.classical_conv(x)  # shape: (B, n_qubits, 14, 14)
        B, C, H, W = x.shape
        # Reshape patches to (B*H*W, C)
        patches = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        # Quantum filter on flattened patches
        features = self.quantum_filter(patches)  # shape: (B*H*W, C)
        # Reshape back to per‑image features
        features = features.view(B, H * W, C).permute(0, 2, 1).contiguous()
        features = features.view(B, -1)  # flatten for FC
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)
