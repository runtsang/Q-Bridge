"""Hybrid quantum estimator that supports both parametric circuits and torchquantum modules.

The estimator extends the lightweight FastBaseEstimator from the QML seed, adding optional shot noise
and the ability to evaluate either a Qiskit QuantumCircuit or a torchquantum QFCModel.  It also
provides the quantum QFCModel from the Quantum‑NAT example for convenience.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable, List

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
import numpy as np


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        *,
        qmodule=None,
    ) -> None:
        if circuit is None and qmodule is None:
            raise ValueError("Either a circuit or a quantum module must be provided")
        self._circuit = circuit
        self._qmodule = qmodule
        if circuit is not None:
            self._parameters = list(circuit.parameters)
        else:
            self._parameters = None  # inferred from qmodule

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if self._circuit is None:
            raise RuntimeError("No circuit to bind")
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            if self._circuit is not None:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Assume the qmodule exposes an evaluate method returning complex expectations.
                row = self._qmodule.evaluate(
                    statevector=True, parameters=values, observables=observables
                )
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional shot noise to the deterministic quantum estimator."""
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
                rng.normal(complex(val.real, val.imag), max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


# Import the quantum QFCModel architecture from the Quantum‑NAT example
from.QuantumNAT import QFCModel as QuantumQFCModel


class FastHybridEstimator(FastEstimator):
    """
    Hybrid estimator that can wrap either a Qiskit circuit or a torchquantum QFCModel.

    Parameters
    ----------
    model : QuantumCircuit | None
        The quantum model. If ``None`` and ``model_type`` is ``'qfc'``, a default
        QuantumQFCModel is instantiated.
    model_type : str, optional
        ``'circuit'`` (default) or ``'qfc'``. Determines the type of model to create
        when ``model`` is ``None``.
    """
    def __init__(
        self,
        model: QuantumCircuit | None = None,
        *,
        model_type: str = "circuit",
        **model_kwargs,
    ) -> None:
        if model is None:
            if model_type == "qfc":
                model = QuantumQFCModel()
            else:
                raise ValueError("model must be provided for circuit type")
        if isinstance(model, QuantumCircuit):
            super().__init__(circuit=model)
        else:
            super().__init__(qmodule=model)


__all__ = ["FastBaseEstimator", "FastEstimator", "FastHybridEstimator"]
