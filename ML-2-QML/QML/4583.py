import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from typing import Iterable, Sequence, List

class HybridFCL:
    """
    Quantum counterpart of the classical HybridFCL.  A parameterized circuit
    with data‑encoding (RX) followed by depth layers of RY and CZ gates.
    The circuit outputs expectation values of Pauli‑Z on each qubit, which
    match the observables of the classical network.
    """
    def __init__(self, n_qubits: int = 1, depth: int = 1, shots: int = 100) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        # Build circuit
        self._encoding = ParameterVector("x", self.n_qubits)
        self._weights = ParameterVector("theta", self.n_qubits * self.depth)
        self.circuit = QuantumCircuit(self.n_qubits)
        for qubit, param in zip(range(self.n_qubits), self._encoding):
            self.circuit.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.n_qubits):
                self.circuit.ry(self._weights[idx], qubit)
                idx += 1
            for qubit in range(self.n_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)
        self.circuit.measure_all()
        # Observables: Pauli-Z on each qubit
        self._observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - i - 1))
            for i in range(self.n_qubits)
        ]

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied flat list of parameters.
        Parameters are bound in the order: encoding first, then variational
        weights.  Returns expectation values for the Pauli‑Z observables.
        """
        total_params = self.n_qubits + self.n_qubits * self.depth
        if len(thetas)!= total_params:
            raise ValueError(f"Expected {total_params} parameters, got {len(thetas)}")
        mapping = {}
        # encoding
        for param, val in zip(self._encoding, thetas[:self.n_qubits]):
            mapping[param] = val
        # variational
        for param, val in zip(self._weights, thetas[self.n_qubits:]):
            mapping[param] = val
        bound = self.circuit.assign_parameters(mapping, inplace=False)
        job = execute(bound, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound)
        probs = np.array([count / self.shots for count in counts.values()])
        states = np.array([int(k, 2) for k in counts.keys()]).astype(float)
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for each observable and parameter set.
        Uses Statevector simulation for exact results (no shots).
        """
        observables = list(observables) or self._observables
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self.circuit.assign_parameters(
                {p: v for p, v in zip(self.circuit.parameters, params)},
                inplace=False,
            )
            sv = Statevector.from_instruction(bound)
            row = [sv.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    @property
    def encoding(self) -> List[int]:
        """Indices of input features that are encoded."""
        return list(range(self.n_qubits))

    @property
    def weight_sizes(self) -> List[int]:
        """Number of variational parameters per depth layer."""
        return [self.n_qubits] * self.depth

    @property
    def observables(self) -> List[SparsePauliOp]:
        """Pauli‑Z observables matching the classical network outputs."""
        return self._observables

class FastEstimator(HybridFCL):
    """
    Adds Gaussian shot noise to the deterministic quantum estimator.
    """
    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(float(v), max(1e-6, 1 / shots)) for v in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridFCL", "FastEstimator"]
