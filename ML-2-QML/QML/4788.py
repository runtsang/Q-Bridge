import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class HybridFCLClassifier:
    """
    Quantum implementation that mirrors the classical HybridFCLClassifier.
    Provides a parameterised circuit with data‑encoding (Rx), variational
    Ry layers, CZ entangling gates, and Z measurement on each qubit.

    Attributes
    ----------
    encoding_indices : list[int]
        Indices of the qubits that receive the data‑encoding parameters.
    weight_sizes : list[int]
        Number of variational parameters per depth layer.
    observables : list[SparsePauliOp]
        Z observable on each qubit, matching the classical output shape.
    """

    def __init__(self, num_qubits: int, depth: int, shots: int = 1024, backend=None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or transpile('qasm_simulator', backend=None).backend

        # Parameter vectors
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Build base circuit
        self.circuit = QuantumCircuit(num_qubits)
        for qubit, param in zip(range(num_qubits), self.encoding):
            self.circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        self.circuit.measure_all()

        # Observables: one Pauli Z per qubit
        self.observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                            for i in range(num_qubits)]

        # Metadata
        self.encoding_indices = list(range(num_qubits))
        self.weight_sizes = [num_qubits] * depth

    def run(self, thetas: np.ndarray, encodings: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with the supplied encoding and variational
        parameters and return the expectation value of each Z observable.

        Parameters
        ----------
        thetas : ndarray, shape (num_qubits * depth,)
            Variational parameters for the Ry gates.
        encodings : ndarray, shape (num_qubits,)
            Encoding parameters for the Rx gates.

        Returns
        -------
        expectation : ndarray, shape (num_qubits,)
            Expectation value of the Z observable on each qubit.
        """
        param_binds = [
            {self.encoding[i]: encodings[i] for i in range(self.num_qubits)},
            {self.weights[j]: thetas[j] for j in range(self.num_qubits * self.depth)}
        ]

        job = execute(
            self.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        expectation = np.zeros(self.num_qubits)
        for state, cnt in counts.items():
            prob = cnt / self.shots
            # state string is ordered from qubit 0 to n-1
            for i, bit in enumerate(reversed(state)):
                expectation[i] += (1 if bit == '0' else -1) * prob
        return expectation
