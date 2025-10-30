"""FastBaseEstimator – Quantum implementation with Aer simulation and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parameter import Parameter
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import BaseOperator
from qiskit.providers.basicaerprovider import BasicAerProvider
from qiskit.providers.exceptions import JobError
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller, BasisTranslator, CXBasisTranslation
from qiskit.visualization import plot_histogram

# Helper to compute expectation via statevector
def _expectation(state: Statevector, observable: BaseOperator) -> complex:
    return state.expectation_value(observable)

class FastBaseEstimator:
    """
    Evaluate expectation values of parameter‑dependent quantum circuits.

    Parameters
    ----------
    circuit : QuantumCircuit
        A parameterised circuit with symbolic parameters.
    noise_model : Optional[Callable[[QuantumCircuit], QuantumCircuit]]
        Function that injects noise into a circuit before execution.
    shots : int | None
        Number of shots for measurement‑based evaluation; if None, use statevector.
    backend : str | AerSimulator, optional
        Backend identifier or instance. Defaults to AerSimulator.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        noise_model: Optional[Callable[[QuantumCircuit], QuantumCircuit]] = None,
        shots: Optional[int] = None,
        backend: str | AerSimulator = "aer_simulator",
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.noise_model = noise_model
        self.shots = shots
        if isinstance(backend, str):
            self.backend = AerSimulator()
            self.backend_name = backend
        else:
            self.backend = backend
            self.backend_name = backend.name

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound."""
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._parameters, param_values))
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
            circ = self._bind(values)
            if self.noise_model:
                circ = self.noise_model(circ)
            if self.shots is None:
                # State‑vector evaluation
                state = Statevector.from_instruction(circ)
                row = [_expectation(state, obs) for obs in observables]
            else:
                # Shot‑based evaluation
                job = self.backend.run(circ, shots=self.shots)
                try:
                    result = job.result()
                except JobError as exc:
                    raise RuntimeError(f"Quantum job failed: {exc}") from exc
                counts = result.get_counts()
                # Convert counts to statevector estimate
                state = Statevector.from_label(counts)
                row = [_expectation(state, obs) for obs in observables]
            results.append(row)
        return results

    def parameter_shift_gradient(
        self,
        observable: BaseOperator,
        parameter_sets: Sequence[Sequence[float]],
        shift: float = np.pi / 2,
    ) -> List[List[complex]]:
        """
        Compute gradients of an observable with respect to all parameters
        using the parameter‑shift rule.

        Returns
        -------
        grads : List[ List[complex] ]
            Outer list over parameter sets, inner list over parameters.
        """
        grads: List[List[complex]] = []

        for values in parameter_sets:
            base_circ = self._bind(values)
            base_expect = self.evaluate([observable], [values])[0][0]
            grad_row: List[complex] = []

            for i, param in enumerate(self._parameters):
                # Shift forward
                shifted_plus = list(values)
                shifted_plus[i] += shift
                circ_plus = self._bind(shifted_plus)
                exp_plus = self.evaluate([observable], [shifted_plus])[0][0]

                # Shift backward
                shifted_minus = list(values)
                shifted_minus[i] -= shift
                circ_minus = self._bind(shifted_minus)
                exp_minus = self.evaluate([observable], [shifted_minus])[0][0]

                # Parameter‑shift gradient
                grad = 0.5 * (exp_plus - exp_minus)
                grad_row.append(grad)
            grads.append(grad_row)
        return grads


__all__ = ["FastBaseEstimator"]
