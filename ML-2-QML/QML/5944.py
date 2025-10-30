"""Quantum quanvolution module and exact estimator.

The module implements the same logical pipeline as the classical
version but replaces the convolutional filter with a
parameter‑dependent quantum kernel.  Qiskit is used for exact
state‑vector simulation, and a dedicated `FastBaseEstimator` provides
expectation‑value evaluation for arbitrary observables.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections.abc import Iterable, Sequence
from typing import Iterable as IterableType, List, Sequence as SequenceType
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QuanvolutionFilterQuantum(nn.Module):
    """Quantum kernel acting on 2×2 image patches.

    A 4‑qubit variational circuit is applied to each patch.  The
    circuit consists of Ry rotations encoding the pixel intensities,
    followed by a fixed random two‑qubit layer that entangles the wires.
    The measurement is performed in the Pauli‑Z basis.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Base circuit with Ry encodings
        self.base_circuit = QuantumCircuit(self.n_wires)
        for i in range(self.n_wires):
            self.base_circuit.ry(0.0, i)  # placeholder angles

        # Random entangling layer
        self.random_layer = QuantumCircuit(self.n_wires)
        self.random_layer.cx(0, 1)
        self.random_layer.cx(1, 2)
        self.random_layer.cx(2, 3)
        self.random_layer.cx(3, 0)

    def _encode(self, circuit: QuantumCircuit, data: torch.Tensor) -> QuantumCircuit:
        """Apply Ry rotations proportional to pixel intensities."""
        for i, val in enumerate(data):
            circuit.ry(val.item(), i)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return concatenated measurement results for all patches."""
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Build circuit for each batch element
                measurements = []
                for batch_idx in range(bsz):
                    qc = self.base_circuit.copy()
                    qc = self._encode(qc, patch[batch_idx])
                    qc = qc.compose(self.random_layer, inplace=False)
                    state = Statevector.from_instruction(qc)
                    meas = state.expectation_value(1)  # Z expectation
                    measurements.append(meas)
                patches.append(torch.tensor(measurements, dtype=torch.float32))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifierQuantum(nn.Module):
    """Quantum classifier mirroring the classical architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


class FastBaseEstimatorQuantum:
    """Exact expectation‑value evaluator for a parameterised quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate.  Parameters must be bound before
        execution.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: IterableType[BaseOperator],
        parameter_sets: SequenceType[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Each observable is a `BaseOperator` instance.  The method returns
        a list of lists containing complex expectation values.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimatorQuantum(FastBaseEstimatorQuantum):
    """Extends the exact evaluator with shot‑noise simulation.

    The noise model samples from a normal distribution with variance
    1/shots, emulating a finite‑sample measurement process.
    """

    def evaluate(
        self,
        observables: IterableType[BaseOperator],
        parameter_sets: SequenceType[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(row_val.real, max(1e-6, 1 / shots)),
                    rng.normal(row_val.imag, max(1e-6, 1 / shots)),
                )
                for row_val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = [
    "QuanvolutionFilterQuantum",
    "QuanvolutionClassifierQuantum",
    "FastBaseEstimatorQuantum",
    "FastEstimatorQuantum",
]
