import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers.aer import AerSimulator

class UnifiedSelfAttention:
    """
    Quantum self‑attention module implemented with Qiskit.
    Each token is encoded in a block of qubits, entangled with its
    neighbours, and measured to produce a vector of expectation
    values that can be used as attention scores or as input to
    a downstream classical readout.
    """
    def __init__(self,
                 n_qubits: int,
                 backend: AerSimulator | None = None,
                 shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

    def _build_circuit(self, token: np.ndarray,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        qc = QuantumCircuit(qr, cr)

        # Encode token into qubit rotations
        for i in range(self.n_qubits):
            ang = token[i] * rotation_params[i]
            qc.ry(ang, qr[i])

        # Entangle neighbouring qubits
        for i in range(self.n_qubits - 1):
            qc.cx(qr[i], qr[i + 1])

        # Additional entanglement based on entangle_params
        for i in range(self.n_qubits - 1):
            qc.rzz(entangle_params[i], qr[i], qr[i + 1])

        qc.measure(qr, cr)
        return qc

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Run the quantum self‑attention circuit for each token in the batch.
        Parameters
        ----------
        rotation_params : np.ndarray
            1‑D array of length n_qubits specifying rotation scaling per qubit.
        entangle_params : np.ndarray
            1‑D array of length n_qubits‑1 specifying RZZ entanglement angles.
        inputs : np.ndarray
            Input data of shape (B, T, D) where D <= n_qubits.
        Returns
        -------
        expectation : np.ndarray
            Expectation values of Pauli‑Z for each qubit,
            shape (B, T, n_qubits).
        """
        B, T, D = inputs.shape
        expectation = np.zeros((B, T, self.n_qubits), dtype=np.float32)

        for b in range(B):
            for t in range(T):
                token = inputs[b, t, :self.n_qubits]
                qc = self._build_circuit(token, rotation_params, entangle_params)
                job = execute(qc, self.backend, shots=self.shots)
                result = job.result()
                counts = result.get_counts(qc)

                exp_vals = np.zeros(self.n_qubits)
                for bitstr, cnt in counts.items():
                    # Convert bitstring to +1/-1 values per qubit
                    bits = np.array([1 if bit == '0' else -1 for bit in bitstr])
                    exp_vals += cnt * bits
                exp_vals /= self.shots
                expectation[b, t, :] = exp_vals

        return expectation

    def __repr__(self):
        return f"<UnifiedSelfAttention quantum n_qubits={self.n_qubits}>"

__all__ = ["UnifiedSelfAttention"]
