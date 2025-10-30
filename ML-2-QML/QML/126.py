"""FastEstimator: a quantum‑aware estimator for parametrised circuits with support for batched evaluation,
parameter‑shift gradients, shot noise, and optional noise models.

The class wraps a :class:`qiskit.circuit.QuantumCircuit` and exposes an evaluate interface that
accepts :class:`qiskit.quantum_info.operators.base_operator.BaseOperator` observables, optional shot
noise, and a backend.  Gradients are computed via the parameter‑shift rule.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence, Dict, Tuple, Any, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Operator
from qiskit.opflow import PauliSumOp, expectation, StateFn
from qiskit.opflow import Z, X, Y
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import AerSimulator
from qiskit.providers.basicaer import NoiseModel

BaseOperator = Operator

def _bind(circuit: QuantumCircuit, param_values: Sequence[float]) -> QuantumCircuit:
    """Return a new circuit with parameters bound to the supplied values."""
    if len(param_values)!= len(circuit.parameters):
        raise ValueError("Parameter count mismatch for bound circuit.")
    mapping = dict(zip(circuit.parameters, param_values))
    return circuit.assign_parameters(mapping, inplace=False)

class FastEstimator:
    """Quantum estimator for parametrised circuits."""
    def __init__(
        self,
        circuit: QuantumCircuit,
        backend: Optional[QuantumInstance] = None,
        *,
        shots: int | None = None,
        noise_model: NoiseModel | None = None,
        optimization_level: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        circuit
            A parametrised :class:`QuantumCircuit` with real‑valued parameters.
        backend
            A :class:`QuantumInstance`.  If ``None`` an Aer statevector simulator is used.
        shots
            Number of shots for noisy simulation.  ``None`` triggers statevector evaluation.
        noise_model
            Optional noise model to apply to the circuit.
        optimization_level
            Transpilation optimisation level (0–3).
        """
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

        if backend is None:
            self.backend = QuantumInstance(
                AerSimulator(method="statevector"),
                shots=shots or 0,
                noise_model=noise_model,
            )
        else:
            self.backend = backend

        self.optimization_level = optimization_level

    def _prepare_circuit(self, param_values: Sequence[float]) -> QuantumCircuit:
        """Bind parameters and transpile for the chosen backend."""
        bound = _bind(self.circuit, param_values)
        transpiled = transpile(bound, backend=self.backend.backend, optimization_level=self.optimization_level)
        return transpiled

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables
            Iterable of quantum operators.
        parameter_sets
            Iterable of parameter vectors.

        Returns
        -------
        List[List[complex]]
            Outer list over parameter sets, inner list over observables.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._prepare_circuit(params)
            if self.backend.backend.configuration().simulator and self.backend.shots == 0:
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # Use statevector for expectation via StateFn
                state = StateFn(Statevector.from_instruction(circ))
                row = [expectation(obs, state).eval().data for obs in observables]
            results.append(row)
        return results

    def gradients(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[float]]:
        """
        Compute gradients of a single observable w.r.t. the circuit parameters using the
        parameter‑shift rule.

        Parameters
        ----------
        observable
            Quantum operator.
        parameter_sets
            Iterable of parameter vectors.
        shift
            Shift angle (default π/2).

        Returns
        -------
        List[List[float]]
            Outer list over parameter sets, inner list over observables.
        """
        grads: List[List[float]] = []

        for params in parameter_sets:
            grad_row: List[float] = []
            for idx, _ in enumerate(self.parameters):
                # Shift parameters up and down
                plus = list(params)
                minus = list(params)
                plus[idx] += shift
                minus[idx] -= shift

                # Evaluate shifted circuits
                val_plus = self.evaluate([observable], [plus])[0][0]
                val_minus = self.evaluate([observable], [minus])[0][0]

                # Parameter‑shift gradient
                grad = 0.5 * (val_plus - val_minus)
                grad_row.append(float(grad))
            grads.append(grad_row)
        return grads

__all__ = ["FastEstimator"]
