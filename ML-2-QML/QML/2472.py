"""Hybrid self‑attention with a quantum fully‑connected sub‑layer."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class HybridSelfAttentionQuantum:
    """
    Quantum self‑attention block that embeds a parameterised fully‑connected
    layer implemented as a single‑qubit circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used for the attention circuit.
    backend : qiskit.providers.Backend
        Quantum backend to execute the circuit.
    shots : int
        Number of shots for measurement.
    """
    def __init__(self, n_qubits: int, backend, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        # Prepare the fully‑connected sub‑circuit (one qubit)
        self._fc_circuit = QuantumCircuit(1, 1)
        self.theta = qiskit.circuit.Parameter("theta")
        self._fc_circuit.h(0)
        self._fc_circuit.ry(self.theta, 0)
        self._fc_circuit.measure(0, 0)

    def _build_attention_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """
        Build the attention circuit from rotation and entangle parameters.
        """
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(qr, cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        thetas: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the hybrid quantum attention and fully‑connected sub‑circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the attention rotation gates.
        entangle_params : np.ndarray
            Parameters for the attention entanglement gates.
        thetas : np.ndarray
            Parameter values for the fully‑connected layer.

        Returns
        -------
        np.ndarray
            Combined expectation value from the fully‑connected sub‑circuit
            weighted by the probability distribution of the attention circuit.
        """
        # Attention circuit
        attn_circuit = self._build_attention_circuit(rotation_params, entangle_params)
        attn_job = execute(attn_circuit, self.backend, shots=self.shots)
        attn_counts = attn_job.result().get_counts(attn_circuit)
        attn_probs = np.array(list(attn_counts.values())) / self.shots

        # Fully‑connected sub‑circuit
        fc_job = execute(
            self._fc_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        fc_counts = fc_job.result().get_counts(self._fc_circuit)
        fc_probs = np.array(list(fc_counts.values())) / self.shots
        # Expectation value of the single‑qubit measurement (0 → 0, 1 → 1)
        fc_expect = np.sum(np.array(list(fc_counts.keys()), dtype=int) * fc_probs)

        # Combine: weight the attention probability mass by the FC expectation
        combined = np.sum(attn_probs) * fc_expect
        return np.array([combined])

def SelfAttention():
    """
    Factory that returns a HybridSelfAttentionQuantum instance configured
    to mirror the original anchor module.
    """
    backend = Aer.get_backend("qasm_simulator")
    return HybridSelfAttentionQuantum(n_qubits=4, backend=backend, shots=1024)

__all__ = ["SelfAttention"]
