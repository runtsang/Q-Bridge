"""Hybrid estimator combining a variational quantum circuit with optional shot‑noise
simulation and a classical convolutional filter. The implementation uses Qiskit
for circuit construction and execution.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Iterable, List, Optional, Sequence

from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


# --------------------------------------------------------------------------- #
#  Classical convolutional filter (identical to the PyTorch version)
# --------------------------------------------------------------------------- #
class ConvFilter:
    """A simple 2‑D convolutional filter implemented with NumPy that mimics the
    quantum quanvolution. It returns a scalar probability in [0,1].
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Random kernel for demonstration; in practice this could be learned
        self.kernel = np.random.randn(kernel_size, kernel_size)

    def __call__(self, data: Sequence[float]) -> float:
        arr = np.array(data, dtype=np.float32).reshape(
            self.kernel_size, self.kernel_size
        )
        conv = np.sum(arr * self.kernel)
        return 1 / (1 + np.exp(-(conv - self.threshold)))  # sigmoid


# --------------------------------------------------------------------------- #
#  Quantum filter circuit
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Variational circuit that implements a quantum filter similar to a
    quanvolution layer. Parameters are set by the data values after a threshold.
    """
    def __init__(
        self,
        kernel_size: int,
        backend: str = "qasm_simulator",
        shots: int = 100,
        threshold: float = 0.5,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        self.circuit = QuantumCircuit(self.n_qubits)
        self.params = [Parameter(f"theta{i}") for i in range(self.n_qubits)]

        for i in range(self.n_qubits):
            self.circuit.rx(self.params[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: Sequence[float]) -> float:
        """Execute the circuit with data‑dependent parameters and return the
        average probability of measuring |1> across all qubits.
        """
        data_arr = np.array(data, dtype=np.float32).reshape(
            self.n_qubits
        )
        param_binds = []
        for val in data_arr:
            bind = {self.params[i]: np.pi if val > self.threshold else 0.0}
            param_binds.append(bind)

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average number of |1> from measurement outcomes
        total_ones = 0
        for bits, freq in counts.items():
            ones = bits.count("1")
            total_ones += ones * freq
        return total_ones / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
#  Hybrid estimator
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """Evaluate a quantum circuit (or a list of circuits) with optional
    shot‑noise simulation and an optional classical convolutional pre‑processor.
    """
    def __init__(
        self,
        circuit: QuantumCircuit | QuanvCircuit,
        conv_filter: Optional[Callable[[Sequence[float]], float]] = None,
    ) -> None:
        # If the user passes a pre‑built QuanvCircuit, use it directly
        self.circuit = circuit
        self.conv_filter = conv_filter

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if isinstance(self.circuit, QuantumCircuit):
            if len(parameter_values)!= len(self.circuit.parameters):
                raise ValueError("Parameter count mismatch for bound circuit.")
            mapping = dict(zip(self.circuit.parameters, parameter_values))
            return self.circuit.assign_parameters(mapping, inplace=False)
        raise TypeError("Unsupported circuit type for binding.")

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
        observables:
            Qiskit operators whose expectation values are measured.
        parameter_sets:
            Iterable of input vectors for the quantum circuit.
        shots:
            If provided, the circuit is executed with the given number of shots
            and results are treated as noisy measurements.
        seed:
            Random seed for reproducibility of the classical noise simulation.

        Returns
        -------
        List[List[complex]]:
            Nested list with one row per parameter set and one column per observable.
        """
        observables = list(observables) or [lambda _: 0]  # placeholder
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed) if shots is not None else None

        for params in parameter_sets:
            # Optional classical pre-processing
            if self.conv_filter is not None:
                preprocessed = self.conv_filter(params)
                data = [preprocessed] * self.circuit.n_qubits
            else:
                data = params

            if isinstance(self.circuit, QuanvCircuit):
                expectation = self.circuit.run(data)
                row = [expectation for _ in observables]
            else:
                # Deterministic Statevector evaluation
                bound_circ = self._bind(data)
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]

            if shots is not None:
                noisy_row = [
                    rng.normal(float(val), max(1e-6, 1 / shots)) for val in row
                ]
                row = noisy_row

            results.append(row)

        return results


__all__ = ["HybridEstimator", "ConvFilter", "QuanvCircuit"]
