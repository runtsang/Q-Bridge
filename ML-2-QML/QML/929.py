"""Quantum estimator utilities built on Qiskit.

This module mirrors the classical implementation but operates on
parameterized quantum circuits.  It introduces:
* ``FastEstimator`` – optional shot noise via a QASM simulator.
* ``HybridEstimator`` – fuses a quantum circuit with a classical PyTorch model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import numpy as np
import torch
from torch import nn
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional shot noise by sampling from a backend simulator."""
    def __init__(self, circuit: QuantumCircuit, backend=None) -> None:
        super().__init__(circuit)
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
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
            noisy_row = [
                complex(
                    rng.normal(val.real, 1 / np.sqrt(shots)),
                    rng.normal(val.imag, 1 / np.sqrt(shots)),
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


class HybridEstimator:
    """Combine a quantum circuit with a classical PyTorch model.

    The quantum part is evaluated with :class:`FastBaseEstimator` and
    the classical part is a PyTorch ``nn.Module``.  The two outputs are
    fused (concatenated or added) and passed through an optional linear
    layer before observables are computed.

    Parameters
    ----------
    quantum_circuit : QuantumCircuit
        Parametrized quantum circuit.
    classical : nn.Module
        PyTorch model that maps classical parameters to a tensor.
    fusion : str
        ``'concat'`` (default) or ``'add'`` to fuse the two outputs.
    fusion_layer : nn.Module | None
        Optional linear layer applied after fusion.
    """
    def __init__(
        self,
        quantum_circuit: QuantumCircuit,
        classical: nn.Module,
        *,
        fusion: str = "concat",
        fusion_layer: nn.Module | None = None,
    ) -> None:
        if fusion not in {"concat", "add"}:
            raise ValueError("fusion must be 'concat' or 'add'")
        self.quantum = FastBaseEstimator(quantum_circuit)
        self.classical = classical
        self.fusion = fusion
        self.fusion_layer = fusion_layer

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate the hybrid model for all parameter sets."""
        results: List[List[complex]] = []
        self.classical.eval()
        with torch.no_grad():
            for params in parameter_sets:
                # Classical output
                cls_out = self.classical(_ensure_batch(params))
                # Quantum output: expectation values of Pauli-Z on each qubit
                q_expect = self.quantum.evaluate(
                    [BaseOperator.from_label("Z") for _ in range(len(self.quantum._params))],
                    [params],
                )[0]
                q_out = torch.as_tensor(q_expect, dtype=torch.float32)
                if self.fusion == "concat":
                    fused = torch.cat([cls_out, q_out], dim=-1)
                else:
                    fused = cls_out + q_out
                if self.fusion_layer is not None:
                    fused = self.fusion_layer(fused)
                # Compute observables on fused vector (treated as a state vector)
                row: List[complex] = []
                for obs in observables:
                    sv = Statevector.from_int(int(torch.argmax(fused).item()), dims=2 ** len(self.quantum._params))
                    val = sv.expectation_value(obs)
                    row.append(val)
                results.append(row)
        return results


__all__ = ["FastBaseEstimator", "FastEstimator", "HybridEstimator"]
