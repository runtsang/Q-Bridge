import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttentionHybrid:
    """
    Quantum self‑attention block that encodes input embeddings as rotation angles
    and uses a variational entangling layer to produce a probability distribution
    over the keys.  The distribution can be interpreted as attention weights.
    """
    def __init__(self, n_qubits: int, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.qreg = QuantumRegister(n_qubits, "q")
        self.creg = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qreg, self.creg)
        # Encode each input embedding element as rotation on a qubit
        for i in range(self.n_qubits):
            angle = rotation_params[i] + inputs[i]  # simple encoding
            qc.rx(angle, i)
            qc.ry(angle, i)
            qc.rz(angle, i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qreg, self.creg)
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray, shots: int = 1024):
        """
        Execute the attention circuit and return measurement counts.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for the single‑qubit gates (length >= n_qubits).
        entangle_params : np.ndarray
            Parameters for the controlled‑R‑X gates (length >= n_qubits-1).
        inputs : np.ndarray
            Input embedding vector to be encoded into rotations (length == n_qubits).
        shots : int, optional
            Number of shots for the simulator (default 1024).

        Returns
        -------
        dict
            Measurement outcome counts representing the attention distribution.
        """
        qc = self._build_circuit(rotation_params, entangle_params, inputs)
        job = execute(qc, self.backend, shots=shots)
        result = job.result()
        return result.get_counts(qc)

__all__ = ["SelfAttentionHybrid"]
