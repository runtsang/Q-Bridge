import qiskit
import numpy as np

class UnifiedSelfAttentionHybrid:
    """Quantum self‑attention block that encodes the attention logits into a
    parameterised Qiskit circuit. The circuit consists of a rotation layer
    followed by a controlled‑rotation entanglement pattern. The expectation
    value of the Z measurement on all qubits is returned as the attention
    weight."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.rotation_params = None
        self.entangle_params = None

    def set_params(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """
        Set the rotation and entanglement parameters for the circuit.
        Parameters must match the expected dimensionality:
        - rotation_params: array of length 3 * n_qubits
        - entangle_params: array of length n_qubits - 1
        """
        if rotation_params.shape[0]!= 3 * self.n_qubits:
            raise ValueError(f"rotation_params must have length {3 * self.n_qubits}")
        if entangle_params.shape[0]!= self.n_qubits - 1:
            raise ValueError(f"entangle_params must have length {self.n_qubits - 1}")
        self.rotation_params = rotation_params
        self.entangle_params = entangle_params

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        # Rotation layer
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3 * i], i)
            qc.ry(self.rotation_params[3 * i + 1], i)
            qc.rz(self.rotation_params[3 * i + 2], i)
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(self, backend, shots: int = 1024):
        """
        Execute the circuit on the provided backend and return the
        expectation value of the Z measurement over all qubits.
        """
        if self.rotation_params is None or self.entangle_params is None:
            raise RuntimeError("Circuit parameters not set. Call set_params first.")
        qc = self._build_circuit()
        job = qiskit.execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        return self._counts_to_expectation(counts)

    @staticmethod
    def _counts_to_expectation(counts: dict) -> float:
        """
        Convert a counts dictionary into the expectation value of a Z
        measurement on a multi‑qubit system.
        """
        total = sum(counts.values())
        exp = 0.0
        for state, count in counts.items():
            z = sum(1 if bit == '0' else -1 for bit in state)
            exp += z * count / total
        return exp

__all__ = ["UnifiedSelfAttentionHybrid"]
