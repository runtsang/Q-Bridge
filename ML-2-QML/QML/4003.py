"""Quantum‑centric estimator utilities using TorchQuantum.

This module provides a quantum variant of FastBaseEstimatorGen that
evaluates expectation values of a parametrized circuit or quantum
module.  It supports shot noise, automatic parameter binding, and
a small regression dataset.

Example
-------
>>> from. import FastBaseEstimatorGen, RegressionDataset, QModel
>>> est = FastBaseEstimatorGen(num_wires=2)
>>> dataset = RegressionDataset(samples=256, num_wires=2)
>>> param_sets = [sample["states"].tolist() for sample in dataset]
>>> obs = [torchquantum.operator.PauliZ()]
>>> est.evaluate(obs, param_sets, shots=1000)
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np
import torch
import torchquantum as tq
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence of parameters into a batch tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimatorGen:
    """Evaluate a TorchQuantum module for batched parameter sets.

    The module may be a parametrized circuit (via ``tq.QuantumCircuit``)
    or a ``tq.QuantumModule`` that can be applied to a device.
    """

    def __init__(self, module: tq.QuantumModule | tq.QuantumCircuit) -> None:
        self.module = module
        if isinstance(module, tq.QuantumCircuit):
            self.n_wires = module.num_wires
        else:
            self.n_wires = getattr(module, "n_wires", None)

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> list[list[complex]]:
        """Return expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : Iterable
            Quantum operators for which to compute expectation values.
        parameter_sets : Sequence[Sequence[float]]
            2‑D iterable of parameters for each evaluation.
        shots : int, optional
            If provided, inject Gaussian noise with variance 1/shots.
        seed : int, optional
            Random seed for reproducible noise.
        """
        observables = list(observables)
        results: list[list[complex]] = []

        for params in parameter_sets:
            # Build a fresh device for each evaluation
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires,
                bsz=1,
                device="cpu",
            )
            # Bind parameters and apply the module
            if isinstance(self.module, tq.QuantumCircuit):
                bound = self.module.assign_parameters(
                    dict(zip(self.module.parameters(), params)), inplace=False
                )
                bound(qdev)
            else:
                mapping = dict(zip(self.module.parameters(), params))
                self.module.assign_parameters(mapping, inplace=False)
                self.module(qdev)

            # Compute expectation values from the resulting statevector
            circuit = qdev.circuit
            state = tq.Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: list[list[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(float(val.real), max(1e-6, 1 / shots))
                + 1j * rng.normal(float(val.imag), max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset of synthetic quantum regression samples."""

    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def generate_superposition_data(
    num_wires: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum states of the form
    cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class QModel(tq.QuantumModule):
    """Quantum regression network built on TorchQuantum."""

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
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
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
    "FastBaseEstimatorGen",
    "RegressionDataset",
    "QModel",
    "generate_superposition_data",
]
