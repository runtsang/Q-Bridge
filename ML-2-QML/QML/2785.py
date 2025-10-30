"""Hybrid estimator combining quantum circuit evaluation with optional QuanvCircuit filtering and shot‑noise simulation.

Features:
* Supports a primary parametrized QuantumCircuit.
* Optional QuanvCircuit filter that preprocesses 2‑D data and returns a probability‑based scalar.
* Configurable backend and shot count.
* Expectation‑value evaluation and parameter‑shift gradients.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class HybridFastEstimator:
    """
    Evaluate expectation values of observables for a parametrized circuit, optionally
    applying a QuanvCircuit filter and adding shot‑noise.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        backend: Optional[qiskit.providers.Backend] = None,
        shots: int = 1024,
        filter_circuit: Optional[QuantumCircuit] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        circuit : QuantumCircuit
            The main parametrized circuit to evaluate.
        backend : qiskit.providers.Backend, optional
            The simulator or hardware backend. Defaults to Aer qasm_simulator.
        shots : int
            Number of shots for each execution. Used for shot‑noise emulation.
        filter_circuit : QuantumCircuit, optional
            A QuanvCircuit that preprocesses 2‑D data and returns a scalar.
        seed : int, optional
            Random seed for shot‑noise simulation.
        """
        self.circuit = circuit
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.filter_circuit = filter_circuit
        self.rng = np.random.default_rng(seed)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _run_filter(self, data: np.ndarray) -> float:
        """
        Run the QuanvCircuit filter on classical data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        if self.filter_circuit is None:
            return 0.0

        n_qubits = self.filter_circuit.num_qubits
        data_flat = data.reshape(1, n_qubits)
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.filter_circuit.parameters[i]] = np.pi if val > 0 else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self.filter_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.filter_circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * n_qubits)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables whose expectation values are computed.
        parameter_sets : sequence of sequences
            Each inner sequence holds the parameters for one evaluation.

        Returns
        -------
        List[List[complex]]
            A list of rows, one per parameter set, each containing the observable values.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if self.shots is None:
            return results

        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(self.rng.normal(x.real, 1 / np.sqrt(self.shots)),
                        self.rng.normal(x.imag, 1 / np.sqrt(self.shots)))
                for x in row
            ]
            noisy.append(noisy_row)
        return noisy

    def parameter_shift_gradient(
        self,
        observable: BaseOperator,
        parameter_set: Sequence[float],
        shift: float = np.pi / 2,
    ) -> List[float]:
        """
        Compute the gradient of an observable w.r.t parameters using the parameter‑shift rule.

        Parameters
        ----------
        observable : BaseOperator
            Observable whose expectation value gradient is computed.
        parameter_set : sequence of floats
            Parameters for which the gradient is computed.
        shift : float, optional
            Shift value (default π/2).

        Returns
        -------
        List[float]
            Gradient vector.
        """
        grads = []
        for idx in range(len(parameter_set)):
            plus = list(parameter_set)
            minus = list(parameter_set)
            plus[idx] += shift
            minus[idx] -= shift
            plus_val = self.evaluate([observable], [plus])[0][0].real
            minus_val = self.evaluate([observable], [minus])[0][0].real
            grads.append((plus_val - minus_val) / (2 * np.sin(shift)))
        return grads

    def __repr__(self) -> str:
        return f"<HybridFastEstimator circuit={self.circuit.name} shots={self.shots}>"

__all__ = ["HybridFastEstimator"]
