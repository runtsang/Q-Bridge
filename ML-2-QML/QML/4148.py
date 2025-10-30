import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer

class HybridLayer:
    """
    Quantum‑parameterized hybrid layer that uses a Qiskit circuit to generate
    attention weights via expectation values.  The circuit is parameterized
    by rotation angles and entangling angles; the resulting expectations are
    linearly mapped to the desired output dimension.
    """

    def __init__(self, input_dim: int, output_dim: int, n_qubits: int | None = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits if n_qubits is not None else input_dim
        self.backend = Aer.get_backend("qasm_simulator")

        # Classical post‑processing weights (learned externally if desired)
        self.post_weights = np.random.randn(output_dim, self.n_qubits)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum circuit to obtain attention weights and map them
        to the output space.

        Args:
            rotation_params: Array of length 3 * n_qubits containing RX, RY, RZ
                             angles for each qubit.
            entangle_params: Array of length n_qubits-1 containing CRX angles.
            inputs: Classical input vector (ignored in the quantum core but kept for
                    API compatibility).
            shots: Number of shots for the simulator.

        Returns:
            Numpy array of shape (output_dim,).
        """
        # Build the parameterized circuit
        circ = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure_all()

        # Execute circuit
        job = execute(circ, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circ)

        # Convert counts to probabilities
        probs = np.array(
            [
                counts.get(f"{state:0{self.n_qubits}b}", 0) / shots
                for state in range(2 ** self.n_qubits)
            ]
        )

        # Compute expectation values for each qubit
        expectations = np.zeros(self.n_qubits)
        for qubit in range(self.n_qubits):
            exp = 0.0
            for state, p in enumerate(probs):
                bit = (state >> qubit) & 1
                exp += (1 if bit == 0 else -1) * p
            expectations[qubit] = exp

        # Classical linear mapping to output
        return self.post_weights @ expectations

__all__ = ["HybridLayer"]
