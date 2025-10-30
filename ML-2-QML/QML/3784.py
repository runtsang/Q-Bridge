"""Hybrid classifier with classical convolution and a real quantum kernel.

This module implements the same public API as the ML version but replaces the
classical surrogate with a true quantum circuit that operates on 2x2 image
patches.  The quantum filter is built with `torchquantum` and can be
executed on GPU or CPU.  A shot‑noise aware estimator is also provided.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from collections.abc import Iterable, Sequence
from typing import Iterable as IterableType, List, Callable, Sequence as SequenceType

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: tq.QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> tq.QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: IterableType[tq.QuantumOperator],
        parameter_sets: SequenceType[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = tq.StateVector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to quantum expectation estimates."""
    def evaluate(
        self,
        observables: IterableType[tq.QuantumOperator],
        parameter_sets: SequenceType[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [complex(rng.normal(r.real, 1 / shots) + 1j * rng.normal(r.imag, 1 / shots))
                         for r in row]
            noisy.append(noisy_row)
        return noisy


class QuantumFeatureExtractor(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2x2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionHybridClassifier(nn.Module):
    """
    Hybrid classifier that combines a classical 2‑D convolution with a real
    quantum kernel applied to 2x2 image patches.  The quantum features are
    concatenated with the classical convolution output before the linear
    head.
    """
    def __init__(self, n_channels: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, 4, kernel_size=2, stride=2)
        self.qf_extractor = QuantumFeatureExtractor()
        self.linear = nn.Linear(4 * 14 * 14 * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)            # shape: [B,4,14,14]
        flat = features.view(features.size(0), -1)
        q_features = self.qf_extractor(flat)
        combined = torch.cat([flat, q_features], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)


__all__ = [
    "QuanvolutionHybridClassifier",
    "FastEstimator",
    "FastBaseEstimator",
    "QuantumFeatureExtractor",
]
