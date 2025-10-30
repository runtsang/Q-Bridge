"""Hybrid fully connected layer + classifier implemented in Qiskit."""

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridFCLClassifier:
    """
    Quantum counterpart of the classical hybrid fully‑connected classifier.

    The circuit construction follows the same layering pattern as the
    classical network: an encoding layer, a sequence of depth layers
    each containing a rotation and a CZ‑coupling, and a set of Pauli‑Z
    observables used to read out the expectation values.
    """
    def __init__(self, n_qubits: int = 1, depth: int = 1,
                 backend: str = "qasm_simulator", shots: int = 100):
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = Aer.get_backend(backend)
        self.shots = shots

        # Build the variational circuit
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.n_qubits)
        weights = ParameterVector("theta", self.n_qubits * self.depth)

        qc = QuantumCircuit(self.n_qubits)

        # Encoding layer
        for param, qubit in zip(encoding, range(self.n_qubits)):
            qc.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.n_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.n_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Measurement
        qc.measure_all()

        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - i - 1))
                       for i in range(self.n_qubits)]

        return qc, list(encoding), list(weights), observables

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied variational parameters."""
        param_binds = [{self.weights[i]: theta for i, theta in enumerate(thetas)}]
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def get_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """Return encoding indices, weight sizes, and observable indices."""
        weight_sizes = [self.n_qubits * self.depth]
        return list(range(self.n_qubits)), weight_sizes, list(range(self.n_qubits))

__all__ = ["HybridFCLClassifier"]
