import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttention:
    """
    Quantum self‑attention block that encodes input embeddings into a
    variational circuit, applies rotation and entanglement layers, and
    measures qubit probabilities to produce attention weights.
    The interface mirrors the classical implementation so that the same
    `run` method can be used with a backend, rotation parameters,
    entanglement parameters, and input data.
    """

    def __init__(self, n_qubits: int = 4, seq_len: int = 4):
        self.n_qubits = n_qubits
        self.seq_len = seq_len
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray,
                       inputs: np.ndarray) -> QuantumCircuit:
        """
        Build a variational circuit that encodes the input embeddings
        into qubit rotations, applies a rotation layer, an entanglement
        layer, and measures each qubit.
        """
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encode each input token as a rotation on a distinct qubit
        for i in range(min(self.seq_len, self.n_qubits)):
            theta = inputs[i] * np.pi  # simple scaling to [0, π]
            circuit.ry(theta, i)

        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.rz(entangle_params[i], i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the quantum attention circuit and return a probability
        distribution over the sequence positions that serves as attention
        weights.

        Parameters
        ----------
        rotation_params, entangle_params : np.ndarray
            Parameters for the variational circuit.
        inputs : np.ndarray
            Input embeddings of shape (seq_len,).
        shots : int, default 1024
            Number of shots for the simulation.

        Returns
        -------
        probs : np.ndarray
            Probability vector of shape (seq_len,) representing attention
            weights over the input sequence.
        """
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        probs = np.zeros(self.seq_len)
        for key, val in counts.items():
            # Each key is a bitstring; interpret the first seq_len bits
            for i, bit in enumerate(key[:self.seq_len]):
                if bit == '1':
                    probs[i] += val
        probs /= shots
        return probs

__all__ = ["SelfAttention"]
