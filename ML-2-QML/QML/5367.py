import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator

class HybridFCL:
    """
    Quantum implementation of the hybrid fully connected layer.
    The circuit encodes a one‑dimensional input vector via Ry rotations,
    applies a trainable variational layer, and measures all qubits.
    The expectation value of the Z measurement is returned as a scalar.
    A simple quantum kernel is exposed via ``kernel_matrix``.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 depth: int = 2,
                 backend: str = "qasm_simulator",
                 shots: int = 512):
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = AerSimulator()
        self._build_circuit()

    def _build_circuit(self):
        self.circuit = QuantumCircuit(self.n_qubits)
        # Parameter vector for encoding and variational layers
        self.params = ParameterVector("theta", self.n_qubits * (1 + self.depth))
        # Encoding layer
        for i in range(self.n_qubits):
            self.circuit.ry(self.params[i], i)
        # Variational layers
        idx = self.n_qubits
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                self.circuit.rx(self.params[idx], q)
                idx += 1
            # Entangling CZ gates
            for q in range(self.n_qubits - 1):
                self.circuit.cz(q, q + 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with a single set of parameters.
        ``thetas`` must match the number of parameters in the circuit.
        """
        param_bind = {self.params[i]: thetas[i] for i in range(len(thetas))}
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_bind])
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        # Compute expectation of Z for all qubits and average
        states = np.array([int(k, 2) for k in result.keys()])
        exp = np.sum([(-1)**bin(s).count("1") * p for s, p in zip(states, probs)])
        return np.array([exp])

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Quantum kernel matrix between two batches using the same circuit.
        For each pair (x, y) the parameters are concatenated as [x, -y].
        """
        n = a.shape[0]
        m = b.shape[0]
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                params = np.concatenate([a[i], -b[j]])
                K[i, j] = self.run(params)[0]
        return K

    @staticmethod
    def generate_data(num_qubits: int, samples: int):
        """
        Generate synthetic quantum regression data (mirroring the seed).
        """
        omega_0 = np.zeros(2 ** num_qubits, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_qubits, dtype=complex)
        omega_1[-1] = 1.0
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_qubits), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels.astype(np.float32)

__all__ = ["HybridFCL"]
