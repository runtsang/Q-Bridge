"""Hybrid estimator that evaluates classical PyTorch models and quantum circuits.

This module implements :class:`FastHybridEstimator`, which can handle either a
PyTorch ``nn.Module`` or a Qiskit ``QuantumCircuit``.  It supports
deterministic evaluation, optional Gaussian shot noise, and factory methods
for a classical Quanvolution classifier as well as a quantum Quanvolution
classifier built with :mod:`torchquantum`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution filter with 1 input channel and 4 output channels."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Simple classifier using the QuanvolutionFilter followed by a linear head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class FastHybridEstimator:
    """Evaluates classical PyTorch models or quantum circuits.

    Parameters
    ----------
    model : nn.Module | QuantumCircuit
        The model to evaluate.  If a :class:`~qiskit.circuit.QuantumCircuit` is
        passed, the estimator automatically falls back to the quantum
        evaluation path.
    """

    def __init__(self, model: Union[nn.Module, "QuantumCircuit"]) -> None:
        self.model = model
        self._is_quantum = hasattr(model, "assign_parameters")

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Evaluate the model on a set of parameters.

        For classical models the observables are callables that receive the
        model output tensor.  For quantum models the observables are expected
        to be Qiskit ``BaseOperator`` instances and the estimator will compute
        expectation values using a Statevector simulator.
        """
        if self._is_quantum:
            return self._evaluate_quantum(observables, parameter_sets, shots, seed)
        return self._evaluate_classical(observables, parameter_sets, shots, seed)

    # ------------------------------------------------------------------ #
    # Classical evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_classical(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int],
        seed: Optional[int],
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

        # Add Gaussian shot noise if requested
        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------ #
    # Quantum evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_quantum(
        self,
        observables: Iterable["BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int],
        seed: Optional[int],
    ) -> List[List[complex]]:
        from qiskit.quantum_info import Statevector
        from qiskit.quantum_info.operators.base_operator import BaseOperator

        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind_quantum(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        # shot noise simulation
        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(float(v.real), max(1e-6, 1 / shots)))
                         + 1j * rng.normal(float(v.imag), max(1e-6, 1 / shots))
                         for v in row]
            noisy.append(noisy_row)
        return noisy

    def _bind_quantum(self, parameter_values: Sequence[float]) -> "QuantumCircuit":
        if not hasattr(self.model, "parameters"):
            raise ValueError("Quantum model does not expose parameters.")
        param_names = list(self.model.parameters)
        if len(parameter_values)!= len(param_names):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(param_names, parameter_values))
        return self.model.assign_parameters(mapping, inplace=False)

    @classmethod
    def create_quanvolution_classifier(cls, num_classes: int = 10) -> "FastHybridEstimator":
        """Convenience constructor for the classical QuanvolutionClassifier."""
        model = QuanvolutionClassifier(num_classes=num_classes)
        return cls(model)

    @classmethod
    def create_quantum_quanvolution_classifier(
        cls, *,
        num_classes: int = 10,
        n_wires: int = 4,
        random_layer_ops: int = 8,
        device: str = "cpu",
    ) -> "FastHybridEstimator":
        """Convenience constructor for a quantum QuanvolutionClassifier.

        The implementation uses torchquantum to build a 2×2 patch encoder and a
        random two‑qubit layer.  The resulting model is a subclass of
        :class:`torchquantum.QuantumModule` and can be wrapped by this estimator.
        """
        import torchquantum as tq

        class QuantumQuanvolutionFilter(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [
                        {"input_idx": [0], "func": "ry", "wires": [0]},
                        {"input_idx": [1], "func": "ry", "wires": [1]},
                        {"input_idx": [2], "func": "ry", "wires": [2]},
                        {"input_idx": [3], "func": "ry", "wires": [3]},
                    ]
                )
                self.q_layer = tq.RandomLayer(n_ops=random_layer_ops, wires=list(range(self.n_wires)))
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                bsz = x.shape[0]
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

        class QuantumQuanvolutionClassifier(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.qfilter = QuantumQuanvolutionFilter()
                self.linear = nn.Linear(4 * 14 * 14, num_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.qfilter(x)
                logits = self.linear(features)
                return F.log_softmax(logits, dim=-1)

        model = QuantumQuanvolutionClassifier()
        return cls(model)

__all__ = ["FastHybridEstimator", "QuanvolutionFilter", "QuanvolutionClassifier"]
