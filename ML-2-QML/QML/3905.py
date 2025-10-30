import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class HybridFCL:
    """
    Quantum implementation of a fully connected layer.

    The circuit consists of a layer of Hadamard gates, followed by
    parameterized rotations on each qubit, and a small entangling
    pattern.  The expectation value of the computational basis
    measurement is returned as a one‑dimensional NumPy array.
    The class is compatible with the classical ``HybridFCL`` in that
    its output dimension matches the number of parameters produced
    by the classical model.
    """

    def __init__(self,
                 n_qubits: int = 4,
                 shots: int = 1024,
                 backend: qiskit.providers.BaseBackend = None) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.n_qubits)
        # Initial Hadamard layer to create superposition
        self.circuit.h(range(self.n_qubits))
        # Parameterized rotation layer (will be bound later)
        self.theta = qiskit.circuit.ParameterVector("theta", self.n_qubits)
        for i, theta in enumerate(self.theta):
            self.circuit.ry(theta, i)
        # Simple entangling pattern
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each row of *params*.

        Parameters
        ----------
        params : np.ndarray
            Shape (batch, n_qubits).  Each row contains the parameters
            for a single circuit execution.

        Returns
        -------
        np.ndarray
            Shape (batch, 1) expectation values.
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)
        expectations = []
        for row in params:
            bound_circ = self.circuit.bind_parameters(
                {self.theta[i]: float(row[i]) for i in range(self.n_qubits)}
            )
            job = execute(bound_circ, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(bound_circ)
            # Convert bitstrings to integer values
            exp = 0.0
            total = 0
            for bitstring, cnt in counts.items():
                val = int(bitstring, 2)
                exp += val * cnt
                total += cnt
            expectations.append(exp / total)
        return np.array(expectations).reshape(-1, 1)
