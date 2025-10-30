import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble, Aer, execute

class SelfAttention:
    """
    Variational quantum self‑attention.
    Builds a circuit with single‑qubit rotations and a
    tunable entangling layer.  The output is an attention
    weight vector derived from the expectation of Pauli‑Z
    on each qubit, providing a quantum‑centric attention
    mechanism.
    """
    def __init__(self, n_qubits: int, entanglement_depth: int = 1):
        self.n_qubits = n_qubits
        self.entanglement_depth = entanglement_depth
        self.backend = Aer.get_backend('qasm_simulator')

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """
        Construct a parameterised circuit.
        rotation_params : np.ndarray
            Shape (3 * n_qubits,) – RX, RY, RZ per qubit.
        entangle_params : np.ndarray
            Shape (entanglement_depth * (n_qubits - 1),) – CZ rotation angles.
        """
        qc = QuantumCircuit(self.n_qubits)
        # Apply single‑qubit rotations
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)

        # Entangling layer (repeated `entanglement_depth` times)
        for d in range(self.entanglement_depth):
            for i in range(self.n_qubits - 1):
                theta = entangle_params[d * (self.n_qubits - 1) + i]
                qc.cx(i, i + 1)
                qc.rz(theta, i + 1)
        return qc

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        """
        Execute the circuit and return a normalised attention vector.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        transpiled = transpile(qc, backend)
        qobj = assemble(transpiled, shots=shots)
        job = backend.run(qobj)
        result = job.result()
        counts = result.get_counts(qc)

        # Convert counts to expectation of Z for each qubit
        expectations = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            prob = cnt / shots
            for q in range(self.n_qubits):
                bit = int(state[::-1][q])  # qiskit state string is reversed
                expectations[q] += (1 if bit == 1 else -1) * prob

        # Normalise to [0,1] to interpret as attention weights
        attn = (expectations - expectations.min()) / (
            expectations.max() - expectations.min() + 1e-8
        )
        return attn

__all__ = ["SelfAttention"]
