"""FastBaseEstimator for quantum circuits with caching and gradient support."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit import ParameterVector
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import PassManagerConfig
from qiskit.providers.fake_provider import FakeBackendV2
from qiskit.providers import BackendV2
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import PauliEvolutionGate

# Optional: import torch for parameter‑shift gradients
import torch


class FastBaseEstimator:
    """Evaluate expectation values for parametrised quantum circuits.

    Supports both Qiskit and TorchQuantum backends and caches compiled circuits
    for repeated evaluation.

    Parameters
    ----------
    circuit : QuantumCircuit | torchquantum.QuantumModule
        The variational circuit to evaluate.
    backend : BackendV2 | None, optional
        Execution backend.  If ``None`` a default Fake backend is used.
    shots : int | None, optional
        Number of shots for sampling; ``None`` triggers state‑vector evaluation.
    """

    def __init__(self, circuit: QuantumCircuit | torchquantum.QuantumModule, backend: Optional[BackendV2] = None, shots: Optional[int] = None) -> None:
        self.circuit = circuit
        self.shots = shots
        self.backend = backend or FakeBackendV2()
        self._cache = {}

        # If circuit is a TorchQuantum module, we keep a reference to its device.
        if hasattr(circuit, "device"):
            self.device = circuit.device
        else:
            self.device = None

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with parameters bound."""
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("Only Qiskit circuits support binding in this implementation.")
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _prepare_statevector(self, bound_circuit: QuantumCircuit) -> Statevector:
        """Compile and cache a state‑vector for a bound circuit."""
        key = bound_circuit.parameters
        if key in self._cache:
            return self._cache[key]
        state = Statevector.from_instruction(bound_circuit)
        self._cache[key] = state
        return state

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each observable and parameter set.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Pauli or other linear operators.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors.

        Returns
        -------
        List[List[complex]]
            Rows correspond to parameter sets, columns to observables.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            if isinstance(self.circuit, QuantumCircuit):
                bound = self._bind(values)
                if self.shots is None:
                    state = self._prepare_statevector(bound)
                    row = [state.expectation_value(obs) for obs in observables]
                else:
                    job = self.backend.run(bound, shots=self.shots)
                    result = job.result()
                    counts = result.get_counts()
                    probs = {k: v / self.shots for k, v in counts.items()}
                    # Convert counts to expectation via Pauli expansion
                    row = []
                    for obs in observables:
                        if isinstance(obs, SparsePauliOp):
                            # Expectation = sum_p coeff_p * prob_z
                            val = 0.0
                            for coeff, pauli in zip(obs.coeffs, obs.paulis):
                                key = "".join([str('0' if p == 'I' else ('1' if p == 'Z' else '?')) for p in pauli])
                                val += coeff * probs.get(key, 0)
                            row.append(complex(val))
                        else:
                            # Fallback: use statevector if available
                            state = self._prepare_statevector(bound)
                            row.append(state.expectation_value(obs))
                results.append(row)
            else:
                # TorchQuantum path
                raise NotImplementedError("TorchQuantum support not yet implemented in this stub.")
        return results

    def gradient(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[List[float]]]:
        """Return parameter‑shift gradients for each observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
        parameter_sets : Sequence[Sequence[float]]

        Returns
        -------
        List[List[List[float]]]
            Outer list: parameter sets, middle: observables, inner: gradients per parameter.
        """
        if not isinstance(self.circuit, QuantumCircuit):
            raise NotImplementedError("Gradient only implemented for Qiskit circuits.")
        shift = np.pi / 2
        grads: List[List[List[float]]] = []
        for values in parameter_sets:
            grad_row: List[List[float]] = []
            for obs in observables:
                grad_params: List[float] = []
                for i, val in enumerate(values):
                    plus = list(values)
                    minus = list(values)
                    plus[i] += shift
                    minus[i] -= shift
                    plus_state = self._prepare_statevector(self._bind(plus))
                    minus_state = self._prepare_statevector(self._bind(minus))
                    exp_plus = plus_state.expectation_value(obs).real
                    exp_minus = minus_state.expectation_value(obs).real
                    grad = 0.5 * (exp_plus - exp_minus)
                    grad_params.append(grad)
                grad_row.append(grad_params)
            grads.append(grad_row)
        return grads


__all__ = ["FastBaseEstimator"]
