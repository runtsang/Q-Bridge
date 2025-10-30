import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Iterable, Sequence, Union

class FCL:
    """
    Quantum implementation of a fully‑connected layer using a parameterized
    circuit. Mirrors the classical API: `run` evaluates a single parameter set,
    while `evaluate` processes batches of parameters and observables.
    """
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend if backend is not None else qiskit.Aer.get_backend("qasm_simulator")

        self._circuit = QuantumCircuit(n_qubits)
        self._theta = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n_qubits)]
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        for qubit, theta in enumerate(self._theta):
            self._circuit.ry(theta, qubit)
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for a single parameter set and return the expectation."""
        if len(thetas)!= self.n_qubits:
            raise ValueError("Parameter count must match number of qubits.")
        param_bind = {theta: val for theta, val in zip(self._theta, thetas)}
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self._circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([float(k) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for a list of observables over multiple
        parameter sets. Optional shot noise is added by re‑executing the
        circuit with the supplied number of shots.
        """
        if shots is not None:
            self.shots = shots

        results: List[List[complex]] = []
        for values in parameter_sets:
            if len(values)!= self.n_qubits:
                raise ValueError("Parameter count mismatch.")
            param_bind = dict(zip(self._theta, values))
            bound_circ = self._circuit.assign_parameters(param_bind, inplace=False)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[complex]] = []
            for row in results:
                noisy_row = [rng.normal(complex(v).real, max(1e-6, 1 / shots)) + 1j * 0 for v in row]
                noisy.append(noisy_row)
            return noisy

        return results

__all__ = ["FCL"]
