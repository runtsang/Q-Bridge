"""Hybrid estimator for quantum circuits and quantum modules with optional shot noise.

The estimator accepts either a Qiskit QuantumCircuit or a torchquantum.QuantumModule.
It evaluates expectation values of given observables over multiple parameter sets.
Shot noise can be added to emulate finite‑shot statistics. A quantum QuanvolutionFilter
is provided to process image‑like data before measurement.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union

import numpy as np
import torch
import torchquantum as tq
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

QuantumObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self) -> None:
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


class FastHybridEstimator:
    """Unified estimator for quantum circuits and quantum modules."""

    def __init__(self, model: Union[QuantumCircuit, tq.QuantumModule], *, filter: tq.QuantumModule | None = None) -> None:
        self.model = model
        self.filter = filter
        self.is_circuit = isinstance(model, QuantumCircuit)

    def evaluate(
        self,
        observables: Iterable[BaseOperator | QuantumObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Evaluate the quantum model for each parameter set and observable.

        Parameters
        ----------
        observables
            For circuits: list of BaseOperator. For quantum modules: list of callables
            that accept a state tensor.
        parameter_sets
            Sequence of parameter vectors to bind to the model.
        shots
            If provided, Gaussian noise with variance 1/shots is added to each
            expectation value to mimic finite‑shot sampling.
        seed
            Random seed for reproducibility of the noise.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            if self.is_circuit:
                state = Statevector.from_instruction(self._bind_circuit(params))
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # QuantumModule forward returns a state tensor
                outputs = self.model(params)
                if self.filter is not None:
                    outputs = self.filter(outputs)
                row = [obs(outputs) if callable(obs) else obs for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [
                    complex(
                        rng.normal(mean.real, max(1e-6, 1 / shots))
                        + 1j * rng.normal(mean.imag, max(1e-6, 1 / shots))
                    )
                    for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results

    def _bind_circuit(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.model.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.model.parameters, parameter_values))
        return self.model.assign_parameters(mapping, inplace=False)


__all__ = ["FastHybridEstimator", "QuantumQuanvolutionFilter"]
