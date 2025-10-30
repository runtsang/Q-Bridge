import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple

class QuantumHybridClassifier:
    """
    Quantum implementation of the hybrid classifier.  The circuit consists
    of an encoding layer followed by a layered ansatz and a set of
    Z‑observables that act as the “classical outputs”.  The method
    ``run`` accepts a list of parameter values for the variational
    angles and returns an expectation value that mimics the FCL
    behaviour from the classical seed.
    """

    def __init__(self, num_qubits: int, depth: int, backend=None, shots: int = 1024):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Measurement observables
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]
        return circuit, list(encoding), list(weights), observables

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with a list of variational angles.
        Returns a single expectation value that emulates the
        classical FCL output.
        """
        param_binds = [{self.weights[i]: theta for i, theta in enumerate(thetas)}]
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["QuantumHybridClassifier"]
