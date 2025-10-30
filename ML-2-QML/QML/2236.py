import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer

class QuantumHybridAutoencoder:
    """
    Quantum-only implementation of a fully‑connected layer. The circuit
    applies a Hadamard to each qubit, a parameterized Ry rotation, and
    measures the Z expectation value of each qubit. The output is a
    vector of expectation values of the same dimensionality as the input.
    """

    def __init__(self, n_qubits: int, backend: str = "qasm_simulator", shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend(backend)
        self._build_circuit()

    def _build_circuit(self):
        self.qr = QuantumRegister(self.n_qubits)
        self.cr = ClassicalRegister(self.n_qubits)
        self.circuit = QuantumCircuit(self.qr, self.cr)
        # Prepare the circuit template
        self.circuit.h(self.qr)
        # Ry gates will be parameterized later
        self.circuit.measure(self.qr, self.cr)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each set of parameters in `thetas`.
        `thetas` should be a 2‑D array of shape (batch, n_qubits).
        Returns a 2‑D array of expectation values of shape (batch, n_qubits).
        """
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        batch_size, n_params = thetas.shape
        assert n_params == self.n_qubits, "Parameter vector length must match n_qubits"

        expectations = np.empty((batch_size, self.n_qubits), dtype=np.float32)

        for i, params in enumerate(thetas):
            circ = self.circuit.copy()
            # Bind Ry parameters
            for q, theta in enumerate(params):
                circ.ry(theta, q)
            job = qiskit.execute(circ, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(circ)
            # Compute expectation value for each qubit
            probs = {k: v / self.shots for k, v in counts.items()}
            for q in range(self.n_qubits):
                exp = 0.0
                for bitstring, p in probs.items():
                    bit = int(bitstring[::-1][q])  # bitstring is little‑endian
                    exp += (1 if bit == 0 else -1) * p
                expectations[i, q] = exp
        return expectations
