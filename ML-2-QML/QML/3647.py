from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Iterable, Sequence

class ConvGen285:
    """Quantum variational filter that mirrors the shape of a classical convolution.

    Parameters
    ----------
    kernel_size : int
        Filter width/height; determines the number of qubits.
    threshold : float
        Classical threshold used to parameterise the rotation angles.
    shots : int
        Number of shots for the simulator.
    backend : qiskit.providers.Backend
        Quantum backend; defaults to the Aer qasm simulator.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 127.0, shots: int = 100,
                 backend: qiskit.providers.Backend | None = None) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray | list) -> float:
        """Execute the filter on a 2‑D data patch and return the mean probability of |1>."""
        data = np.reshape(np.asarray(data), (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {theta: (np.pi if val > self.threshold else 0) for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)

        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)

        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += freq * sum(int(b) for b in bitstring)
        return total_ones / (self.shots * self.n_qubits)


class FastConvEstimator:
    """Fast estimator for :class:`ConvGen285` that evaluates expectation values of
    observables over batches of parameter sets.
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
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables
            List of BaseOperator objects whose expectation values are to be computed.
        parameter_sets
            Each inner sequence contains a full set of angles for the circuit.
        shots
            Optional shot‑noise simulation; if ``None`` deterministic Statevector
            evaluation is used.
        seed
            Random seed for Gaussian noise simulation.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots)) for val in row]
                noisy.append(noisy_row)
            return noisy
        return results


__all__ = ["ConvGen285", "FastConvEstimator"]
