"""Quantum self‑attention module using Qiskit.

The circuit applies a parameterised rotation on each qubit followed by a chain of CRX
entangling gates.  Parameters are clipped to a user‑supplied bound to avoid
unphysical gate angles.  The circuit is executed on the Aer qasm simulator and
returns a probability distribution over measurement outcomes.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

__all__ = ["SelfAttentionHybrid"]

class SelfAttentionHybrid:
    """Quantum self‑attention with parameter clipping and CRX entanglement."""
    def __init__(self, n_qubits: int, clip_bounds: float = 5.0) -> None:
        self.n_qubits = n_qubits
        self.clip_bounds = clip_bounds
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _clip(self, params: np.ndarray) -> np.ndarray:
        return np.clip(params, -self.clip_bounds, self.clip_bounds)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        rotation_params = self._clip(rotation_params)
        entangle_params = self._clip(entangle_params)
        circuit = QuantumCircuit(self.qr, self.cr)

        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        """Execute the attention circuit and return outcome probabilities."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = {k: v / shots for k, v in counts.items()}
        return probs
