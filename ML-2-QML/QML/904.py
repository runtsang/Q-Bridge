"""Quantum self‑attention block built on Qiskit.

The circuit encodes a classical attention matrix into a quantum state
using a simple rotation‑based scheme, then applies a variational
layer that refines the distribution.  The implementation is a
drop‑in replacement for the original SelfAttention helper but now
supports amplitude encoding and a configurable depth.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend
from typing import Optional

class SelfAttention__gen181:
    """
    Variational quantum self‑attention.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits used to encode the attention vector.
    depth : int, default 2
        Depth of the variational layer (number of rotation–entanglement
        blocks).
    """

    def __init__(self, n_qubits: int = 4, depth: int = 2):
        self.n_qubits = n_qubits
        self.depth = depth
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _encode_attention(self, attention: np.ndarray) -> QuantumCircuit:
        """
        Encode a flattened attention vector into amplitudes via
        successive Ry rotations.  The method is a lightweight
        approximation suitable for small qubit counts.
        """
        circuit = QuantumCircuit(self.qr)
        flat = attention.flatten()
        norm = np.linalg.norm(flat)
        if norm == 0:
            raise ValueError("Attention vector cannot be all zeros.")
        flat = flat / norm
        for i, amp in enumerate(flat):
            theta = 2 * np.arccos(amp)
            circuit.ry(theta, i)
        return circuit

    def _build_variational(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Construct a variational circuit with the given parameters.
        """
        circuit = QuantumCircuit(self.qr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangling layer repeated 'depth' times
        for _ in range(self.depth):
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
                circuit.rz(entangle_params[i], i + 1)
        return circuit

    def run(
        self,
        backend: Backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        attention_map: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Execution backend.
        rotation_params : np.ndarray
            Shape (3 * n_qubits,) – rotation angles.
        entangle_params : np.ndarray
            Shape (n_qubits - 1,) – entanglement angles.
        attention_map : np.ndarray
            Classical attention matrix (batch, seq_len, seq_len).
        shots : int
            Number of shots.

        Returns
        -------
        refined_attention : np.ndarray
            Refined attention probabilities of the same shape as the input.
        """
        # Encode the attention into a state
        enc_circ = self._encode_attention(attention_map)
        var_circ = self._build_variational(rotation_params, entangle_params)
        circuit = enc_circ + var_circ
        circuit.measure_all()
        job = qiskit.execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)
            probs[idx] = cnt / shots
        refined = probs.reshape(attention_map.shape)
        return refined

def SelfAttention() -> SelfAttention__gen181:
    """
    Factory mirroring the original seed's interface.
    """
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return SelfAttention__gen181(n_qubits=4)

__all__ = ["SelfAttention__gen181", "SelfAttention"]
