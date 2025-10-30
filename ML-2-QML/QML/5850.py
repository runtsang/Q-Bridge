"""Hybrid fast estimator for quantum circuits."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit.primitives import Estimator as BaseEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN

class FastBaseEstimator:
    """Evaluate expectation values of parametric quantum circuits.

    Supports both state‑vector evaluation for exact results and a simulator
    based on Aer for shot‑noise sampling.  The interface mirrors the
    classical FastBaseEstimator for consistency.
    """

    def __init__(self, circuit: QuantumCircuit, simulator: Optional[AerSimulator] = None) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.simulator = simulator or AerSimulator(method="statevector")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return a matrix of expectation values.

        Parameters
        ----------
        observables: iterable of Pauli operators
            Each observable will be evaluated on the circuit state.
        parameter_sets: sequence of sequences
            Parameter values for each circuit instance.
        shots: optional int
            If provided, expectation values are sampled from a state‑vector
            simulator with the given shot count, producing statistical noise.
        seed: optional int
            Random seed for the simulator.
        """
        if parameter_sets is None:
            parameter_sets = []
        if observables is None:
            observables = []

        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                est = BaseEstimator()
                row = []
                for obs in observables:
                    exp_val = est.run(
                        circuits=[bound],
                        observables=[obs],
                        parameter_values=[values],
                        shots=shots,
                        seed_simulator=seed,
                    ).result().values[0]
                    row.append(exp_val)
            results.append(row)

        return results

def EstimatorQNN() -> QEstimatorQNN:
    """Return a toy QNN estimator wrapped around a single‑qubit circuit."""
    params = [Parameter("x"), Parameter("w")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)

    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = BaseEstimator()
    return QEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[params[0]],
        weight_params=[params[1]],
        estimator=estimator,
    )

__all__ = ["FastBaseEstimator", "EstimatorQNN"]
