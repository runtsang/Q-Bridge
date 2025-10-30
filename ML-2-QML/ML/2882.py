"""Hybrid classical regression module with noise-augmented estimator.

This module provides:
* A dataset that samples both classical features and corresponding quantum state
  representations derived from a superposition of |0...0> and |1...1>.
* A feed‑forward network with residual blocks and dropout that learns to map
  classical features to a scalar target.
* A lightweight estimator that can evaluate the network on arbitrary batches of
  parameters and optionally inject Gaussian shot noise, mirroring the FastEstimator
  pattern from the original repo.

The design deliberately mirrors the quantum side: the dataset and estimator
interfaces match those used by the quantum implementation, facilitating
cross‑validation experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Sequence, Iterable, List, Callable, Tuple

# --------------------------------------------------------------------------- #
# Data generation utilities
# --------------------------------------------------------------------------- #

def _generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    The classical features are the angles (theta, phi) used to construct the state.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    # Classical features: [theta, phi] per sample
    features = np.stack([thetas, phis], axis=1).astype(np.float32)
    # Labels: sin(2*theta) * cos(phi)
    labels = np.sin(2 * thetas) * np.cos(phis)
    return features, labels.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class HybridRegressionDataset(Dataset):
    """
    Dataset that returns a tuple of classical features, quantum state vectors,
    and the target scalar.  The quantum state is provided as a complex tensor
    suitable for torchquantum or qiskit usage.
    """

    def __init__(self, samples: int, num_features: int = 2, num_wires: int = 3):
        self.features, self.labels = _generate_superposition_data(num_features, samples)
        self.num_wires = num_wires
        self.states = self._to_state_vector(self.features)

    def _to_state_vector(self, features: np.ndarray) -> np.ndarray:
        """
        Convert (theta, phi) pairs into a 2^n‑dimensional state vector.
        """
        samples = features.shape[0]
        omega_0 = np.zeros(2 ** self.num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** self.num_wires, dtype=complex)
        omega_1[-1] = 1.0
        states = np.empty((samples, 2 ** self.num_wires), dtype=complex)
        for i, (theta, phi) in enumerate(features):
            states[i] = np.cos(theta) * omega_0 + np.exp(1j * phi) * np.sin(theta) * omega_1
        return states.astype(np.complex64)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[index], dtype=torch.float32),
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class HybridRegression(nn.Module):
    """
    Feed‑forward network with residual connections and dropout.

    The architecture is deliberately over‑parameterised to provide a
    meaningful baseline for regression tasks; it can be easily swapped
    with a quantum model that shares the same dataset interface.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Residual connection: add input to output if dimensions match
        self.residual = nn.Identity() if input_dim == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual is not None:
            out = out + self.residual(x)
        return out.squeeze(-1)

# --------------------------------------------------------------------------- #
# Estimator utilities
# --------------------------------------------------------------------------- #

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    """
    Evaluate a PyTorch model on a batch of parameter vectors.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.  The model must accept a single tensor of shape
        (batch_size, input_dim) and return a tensor of shape (batch_size, 1).
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
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
    """
    Adds optional Gaussian shot noise to the deterministic estimator.

    The noise model mimics a finite‑shot measurement by adding zero‑mean Gaussian
    perturbations whose variance scales as 1/shots.
    """

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridRegression", "HybridRegressionDataset", "FastEstimator"]
