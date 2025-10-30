import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class HybridSelfAttentionRegressor:
    """
    Quantum‑centric implementation of the hybrid architecture.  The circuit
    encodes the input image into rotation angles, applies a parameterized
    attention‑style entangling block, a quanvolution‑like two‑qubit kernel,
    and a final measurement that is linearly combined to produce logits.
    """

    def __init__(self,
                 n_qubits: int = 16,
                 n_params: int = 32,
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        # Random parameters for the variational circuit
        self.params = np.random.uniform(0, 2 * np.pi, size=n_params)

    def _build_circuit(self, image: np.ndarray) -> QuantumCircuit:
        """
        Build a quantum circuit that encodes a flattened image patch,
        applies a parameterized attention‑style block and a random
        two‑qubit entangling layer, then measures all qubits.
        """
        # Use the first n_qubits pixel values as rotation angles
        flat = image.flatten()[:self.n_qubits]
        qc = QuantumCircuit(self.n_qubits)
        # Encode pixel values
        for i, val in enumerate(flat):
            qc.ry(val, i)
        # Parameterized gates (attention‑style)
        for i in range(self.n_params):
            qubit = i % self.n_qubits
            qc.rx(self.params[i], qubit)
        # Random two‑qubit entangling layer (quanvolution kernel)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit on a batch of images and return logits.
        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, 1, 28, 28) with pixel values in [0, π].
        Returns
        -------
        np.ndarray
            Logits of shape (batch, 1).
        """
        logits = []
        for img in inputs:
            qc = self._build_circuit(img)
            job = execute(qc, self.backend, shots=self.shots)
            counts = job.result().get_counts(qc)
            # Compute expectation values of Z for each qubit
            exp_vals = np.zeros(self.n_qubits)
            for bitstring, count in counts.items():
                prob = count / self.shots
                for i, bit in enumerate(bitstring[::-1]):  # least significant bit first
                    exp_vals[i] += prob * (1 if bit == "0" else -1)
            # Linear combination to produce a single logit
            weights = np.random.randn(self.n_qubits)
            logits.append(np.dot(exp_vals, weights))
        return np.array(logits).reshape(-1, 1)
