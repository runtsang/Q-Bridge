"""Hybrid estimator for quantum circuits with optional Gaussian shot‑noise."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from typing import Iterable, List, Sequence, Optional

class HybridEstimator:
    """Estimator that can evaluate a Qiskit circuit with optional Gaussian noise
    and a classical post‑processing layer.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized quantum circuit.
    shots : int | None
        Number of shots for simulation. If ``None``, deterministic expectation values
        are returned.
    seed : int | None
        Random seed for Gaussian noise simulation.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("circuit must be a qiskit.QuantumCircuit instance.")
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._backend = Aer.get_backend("aer_simulator")

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def _simulate(self, bound: QuantumCircuit) -> Statevector:
        job = execute(bound, self._backend)
        result = job.result()
        return Statevector.from_dict(result.get_statevector(bound))

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
        *,
        add_shot_noise: bool = False,
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self._bind(params)
            if self.shots is None:
                state = self._simulate(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(bound, self._backend, shots=self.shots)
                result = job.result()
                counts = result.get_counts(bound)
                probs = np.array(list(counts.values())) / self.shots
                bits = np.array([int(k, 2) for k in counts.keys()])
                exp = np.sum(bits * probs)
                row = [exp for _ in observables]  # identical placeholder
            results.append(row)

        if not add_shot_noise or self.shots is None:
            return results

        noisy = []
        for row in results:
            noisy_row = [
                float(self._rng.normal(val.real, max(1e-6, 1 / self.shots))) for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def FCL(n_qubits: int = 1) -> QuantumCircuit:
        """Return a simple parameterized circuit mimicking a fully‑connected layer."""
        theta = qiskit.circuit.Parameter("theta")
        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        qc.ry(theta, range(n_qubits))
        qc.measure_all()
        return qc
