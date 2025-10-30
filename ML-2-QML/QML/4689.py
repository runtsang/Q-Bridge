"""Hybrid quantum sampler that incorporates a parameterised circuit and a fast estimator.

The implementation mirrors the classical sampler but replaces the neural network
with a quantum circuit.  It exposes a factory :func:`SamplerQNN` that returns a
``QuantumSampler`` instance.  The sampler integrates a simple quanvolution
circuit (from the reference) and a :class:`FastBaseEstimator` that evaluates
expectation values on a chosen backend.

The design follows the reference ``SamplerQNN.py`` but extends it by adding
parameter binding logic and shot‑noise simulation.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence

from qiskit import QuantumCircuit, execute, Aer, random
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info.operators.base_operator import BaseOperator


# --------------------------------------------------------------------------- #
# 1. Fast estimator for quantum circuits
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, parameter_values))
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
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# --------------------------------------------------------------------------- #
# 2. Quantum quanvolution filter (from reference 2)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Filter circuit used for quanvolution layers."""

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the circuit on classical data and return average |1> probability."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
# 3. Quantum sampler network
# --------------------------------------------------------------------------- #
class QuantumSampler:
    """Variational sampler that uses a parameterised quantum circuit and a quanvolution filter."""

    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        # Build the sampler circuit
        self.circuit = QuantumCircuit(2)
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        self.circuit.ry(inputs[0], 0)
        self.circuit.ry(inputs[1], 1)
        self.circuit.cx(0, 1)
        for w in weights:
            self.circuit.ry(w, 0 if w is weights[0] else 1)
        self.circuit.cx(0, 1)

        self.estimator = FastBaseEstimator(self.circuit)

        # Quanvolution filter
        self.filter = QuanvCircuit(
            kernel_size=kernel_size,
            backend=Aer.get_backend("qasm_simulator"),
            shots=shots,
            threshold=threshold,
        )
        self.shots = shots

    def run(
        self,
        inputs: Sequence[float],
        image: np.ndarray,
        observables: Iterable[BaseOperator],
    ) -> List[complex]:
        """
        Evaluate the sampler circuit and the quanvolution filter.

        Parameters
        ----------
        inputs : Sequence[float]
            Parameters for the sampler circuit (length 2).
        image : np.ndarray
            2‑D array matching the kernel_size of the quanvolution filter.
        observables : Iterable[BaseOperator]
            Observables for the sampler circuit.

        Returns
        -------
        List[complex]
            Expectation values for the sampler circuit concatenated with the
            filter output as a single observable.
        """
        # Sampler expectation values
        sampler_vals = self.estimator.evaluate(
            observables, [inputs]
        )[0]  # list of lists -> first row

        # Filter output
        filter_out = self.filter.run(image)

        # Combine into a single list
        return sampler_vals + [filter_out]


def SamplerQNN() -> QuantumSampler:
    """Factory that returns a hybrid quantum sampler."""
    return QuantumSampler()


__all__ = ["SamplerQNN", "QuantumSampler", "QuanvCircuit", "FastBaseEstimator"]
