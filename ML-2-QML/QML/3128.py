"""UnifiedFastEstimator for quantum models using TorchQuantum.

This module defines the UnifiedFastEstimator class that can evaluate a
TorchQuantum QuantumModule on batches of input parameters, compute
expectation values of observables, and optionally add Gaussian shot noise.
It also provides helper functions for generating quantum regression data
and a simple TorchDataset for quantum states.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
import torchquantum as tq
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a sequence of floats to a 2‑D batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class UnifiedFastEstimator:
    """Evaluate a TorchQuantum quantum model for a batch of parameters.

    Parameters
    ----------
    model : tq.QuantumModule
        A TorchQuantum quantum model that implements a forward method.
    """

    def __init__(self, model: tq.QuantumModule) -> None:
        self.model = model
        self.n_wires = getattr(model, "n_wires", None)
        if self.n_wires is None:
            if hasattr(model, "encoder"):
                self.n_wires = getattr(model.encoder, "n_wires", None)
            if self.n_wires is None:
                raise ValueError("Unable to determine number of wires for the quantum model.")

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the measurement tensor and returns a
            scalar (e.g., a linear combination of PauliZ measurements).
        parameter_sets : sequence of sequences
            Each inner sequence contains the parameter values for one
            evaluation.  The length must match the number of trainable
            parameters in the model.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to
            each deterministic result to mimic shot noise.
        seed : int, optional
            Random seed for reproducible noise generation.

        Returns
        -------
        List[List[float]]
            Outer list corresponds to the parameter sets; inner list
            contains the value of each observable.
        """
        param_count = sum(1 for _ in self.model.parameters())
        for values in parameter_sets:
            if len(values)!= param_count:
                raise ValueError("Parameter count mismatch for bound circuit.")

        results: List[List[float]] = []

        for values in parameter_sets:
            bsz = len(values)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device="cpu")

            for param, val in zip(self.model.parameters(), values):
                param.data.fill_(val)

            self.model(qdev)

            if hasattr(self.model, "measure") and callable(self.model.measure):
                measurements = self.model.measure(qdev)
            else:
                measurements = qdev.measure_all()

            row = [obs(measurements) if callable(obs) else float(measurements.mean().item()) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy_results: List[List[float]] = []
        for row in results:
            noisy_row = [rng.normal(float(mean), max(1e-6, 1 / shots)) for mean in row]
            noisy_results.append(noisy_row)
        return noisy_results


def generate_quantum_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form
    cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.

    Returns
    -------
    states : ndarray of shape (samples, 2**num_wires)
        Complex amplitude representation of each state.
    labels : ndarray of shape (samples,)
        Target values for regression.
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


class QuantumRegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns quantum states and targets for regression."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_quantum_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Variational quantum circuit for regression."""

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

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = [
    "UnifiedFastEstimator",
    "generate_quantum_data",
    "QuantumRegressionDataset",
    "QModel",
]
