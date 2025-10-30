"""Hybrid quantum estimator that extends FastBaseEstimator with a parameterised circuit
and built‑in shot‑noise simulation. It supports multiple observables and can run on
Qiskit backends."""
import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Optional

def FCL(n_qubits: int = 1, shots: int = 100, backend: Optional[qiskit.providers.Backend] = None) -> QuantumCircuit:
    """Return a simple parameterised circuit used as a toy quantum fully‑connected layer."""
    if backend is None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
    qc = qiskit.QuantumCircuit(n_qubits)
    theta = qiskit.circuit.Parameter("theta")
    qc.h(range(n_qubits))
    qc.ry(theta, range(n_qubits))
    qc.measure_all()
    qc.backend = backend
    qc.shots = shots
    return qc

class FastBaseEstimatorGen326:
    """Quantum estimator that evaluates expectation values for a batch of parameters
    using a parameterised circuit. Optional shot‑noise can be introduced by
    executing the circuit on a backend with a finite number of shots."""
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
        """Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators for which expectation values are requested.
        parameter_sets : sequence of parameter sequences
            Each inner sequence corresponds to one evaluation.
        shots : int, optional
            If supplied, the circuit is executed on the backend with this many shots
            and the result is perturbed by the finite‑sampling noise.
        seed : int, optional
            Random seed for reproducibility of shot‑noise simulation.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        backend = self._circuit.backend
        rng = np.random.default_rng(seed)
        noisy_results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            job = qiskit.execute(
                circ,
                backend=backend,
                shots=shots,
                seed_simulator=seed,
            )
            result = job.result()
            counts = result.get_counts(circ)
            probs = np.array(list(counts.values())) / shots
            bitstrings = np.array([int(b, 2) for b in counts.keys()], dtype=int)
            # For simplicity, map 0->+1, 1->-1 eigenvalues for a PauliZ measurement.
            eigenvals = 1 - 2 * bitstrings
            row: List[complex] = []
            for obs in observables:
                exp_sv = Statevector.from_instruction(self._bind(values)).expectation_value(obs)
                var = exp_sv * (1 - exp_sv)  # crude variance estimate
                noise = rng.normal(0, np.sqrt(var / shots))
                row.append(exp_sv + noise)
            noisy_results.append(row)
        return noisy_results

__all__ = ["FastBaseEstimatorGen326"]
