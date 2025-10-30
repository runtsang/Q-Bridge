import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

class FCL:
    """Parameterized quantum circuit that mimics a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024) -> None:
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots
        self._circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Return the expectation value of the measured bitstring treated as a real number."""
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys()), dtype=float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._circuit.parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Batched evaluation of quantum observables.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Operators whose expectation values are to be computed.  If ``None`` a
            single Pauli‑Z on the first qubit is used.
        parameter_sets : Sequence[Sequence[float]]
            A list of parameter vectors.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added to each
            expectation value.
        seed : int, optional
            Random seed for reproducible noise.

        Returns
        -------
        List[List[complex]]
            A 2‑D list where each row corresponds to a parameter set and each
            column to an observable.
        """
        if parameter_sets is None:
            return []

        observables = list(observables) or [qiskit.quantum_info.Operator.from_label("Z")]
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(val.real, max(1e-6, 1 / shots))) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                         for val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["FCL"]
