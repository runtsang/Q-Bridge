"""Quantum hybrid estimator that evaluates quantum circuits and torchquantum modules.

The estimator provides a unified interface for both Qiskit circuits and
torchquantum modules, includes shot‑noise simulation, and offers factory
methods for a quantum Quanvolution classifier.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchquantum as tq
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum 2×2 patch encoder with a random two‑qubit layer."""

    def __init__(self, n_wires: int = 4, random_layer_ops: int = 8, device: str = "cpu"):
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
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=self.device)
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
    """Hybrid model that applies a quantum quanvolution filter then a linear head."""

    def __init__(
        self, num_classes: int = 10, n_wires: int = 4, random_layer_ops: int = 8, device: str = "cpu"
    ):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter(
            n_wires=n_wires, random_layer_ops=random_layer_ops, device=device
        )
        self.linear = torch.nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class FastHybridEstimator:
    """Quantum estimator for variational circuits and quantum modules.

    Parameters
    ----------
    circuit : QuantumCircuit | tq.QuantumModule
        A parameterised circuit or a torchquantum module.  The estimator
        automatically detects the type and runs the appropriate evaluation
        pipeline.
    """

    def __init__(self, circuit: QuantumCircuit | tq.QuantumModule) -> None:
        self.circuit = circuit
        self._is_quantum_module = isinstance(circuit, tq.QuantumModule)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        For a raw :class:`~qiskit.circuit.QuantumCircuit` the observables are
        Qiskit ``BaseOperator`` instances and evaluation is performed with a
        :class:`~qiskit.quantum_info.Statevector`.  For a
        :class:`~torchquantum.QuantumModule` the circuit is executed on the
        :class:`~torchquantum.QuantumDevice` and the observables are applied
        as ``torchquantum.MeasureAll`` operators.
        """
        if self._is_quantum_module:
            return self._evaluate_torchquantum(observables, parameter_sets, shots, seed)
        return self._evaluate_qiskit(observables, parameter_sets, shots, seed)

    # ------------------------------------------------------------------ #
    # Qiskit evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_qiskit(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int],
        seed: Optional[int],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind_qiskit(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

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

    def _bind_qiskit(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if not hasattr(self.circuit, "parameters"):
            raise ValueError("Circuit does not expose parameters.")
        param_names = list(self.circuit.parameters)
        if len(parameter_values)!= len(param_names):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(param_names, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    # ------------------------------------------------------------------ #
    # TorchQuantum evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_torchquantum(
        self,
        observables: Iterable["tq.MeasureAll"],
        parameter_sets: Sequence[Sequence[float]],
        shots: Optional[int],
        seed: Optional[int],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_module = self._bind_torchquantum(values)
            output = bound_module()
            # Treat each element of the output tensor as a complex amplitude.
            row = [c.item() for c in output.squeeze()]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(float(v.real), max(1e-6, 1 / shots))
                             + 1j * rng.normal(float(v.imag), max(1e-6, 1 / shots))
                             ) for v in row]
            noisy.append(noisy_row)
        return noisy

    def _bind_torchquantum(self, parameter_values: Sequence[float]) -> tq.QuantumModule:
        param_names = list(self.circuit.parameters)
        if len(parameter_values)!= len(param_names):
            raise ValueError("Parameter count mismatch for bound module.")
        for name, val in zip(param_names, parameter_values):
            setattr(self.circuit, name, val)
        return self.circuit

    @classmethod
    def create_quantum_quanvolution_classifier(
        cls,
        *,
        num_classes: int = 10,
        n_wires: int = 4,
        random_layer_ops: int = 8,
        device: str = "cpu",
    ) -> "FastHybridEstimator":
        """Convenience constructor for the quantum QuanvolutionClassifier."""
        model = QuantumQuanvolutionClassifier(
            num_classes=num_classes,
            n_wires=n_wires,
            random_layer_ops=random_layer_ops,
            device=device,
        )
        return cls(model)

__all__ = ["FastHybridEstimator", "QuantumQuanvolutionFilter", "QuantumQuanvolutionClassifier"]
