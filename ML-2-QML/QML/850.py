"""Enhanced FastBaseEstimator for Qiskit circuits with shot‑noise simulation.

The class can evaluate expectation values of arbitrary BaseOperator observables
for a parametrised circuit.  It supports both state‑vector and qasm simulators
and can optionally add shot‑noise to emulate realistic measurements.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised Qiskit circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate.
    backend : str | AerSimulator, optional
        Backend used for simulation.  Defaults to the Aer state‑vector simulator.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: str | AerSimulator | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        if backend is None:
            self._backend = Aer.get_backend("statevector_simulator")
        else:
            self._backend = (
                backend
                if isinstance(backend, AerSimulator)
                else Aer.get_backend(backend)
            )

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable
            BaseOperator instances whose expectation values are desired.
        parameter_sets : sequence of sequences
            Parameter vectors to bind to the circuit.
        shots : int | None, optional
            If provided, use the QASM simulator to generate measurement counts
            and compute expectation values with shot noise.
        seed : int | None, optional
            Random seed for the QASM simulator.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            # State‑vector simulation – noiseless
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

        # Shot‑noise simulation using the QASM simulator
        if not isinstance(self._backend, AerSimulator):
            self._backend = AerSimulator()
        self._backend.set_options(seed_simulator=seed)

        for values in parameter_sets:
            circ = self._bind(values)
            job = self._backend.run(circ, shots=shots)
            result = job.result()
            counts = result.get_counts(circ)
            row = []
            for obs in observables:
                exp_val = 0.0
                for outcome, count in counts.items():
                    prob = count / shots
                    sv = Statevector.from_label(outcome)
                    exp_val += prob * sv.expectation_value(obs)
                row.append(complex(exp_val))
            results.append(row)
        return results


__all__ = ["FastBaseEstimator"]
