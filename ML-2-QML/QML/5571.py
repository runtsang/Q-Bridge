"""Quantum self‑attention implementation for the HybridSelfAttention class.

The quantum version mirrors the classical API but uses a Qiskit
circuit to generate attention‑like weights.  The circuit consists
of a per‑qubit rotation block followed by a chain of controlled‑RX
gates that emulate the entanglement stage of the seed.  The output
is the probability distribution over the computational basis,
which can be interpreted as attention scores.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class HybridSelfAttention:
    """Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits per token; the circuit is built for a
        single token, so the dimensionality of the input must match
        ``n_qubits``.
    mode : str, default 'quantum'
        Mode is fixed to 'quantum' to emphasise that this class
        implements the quantum side of the hybrid layer.
    """
    def __init__(self, n_qubits: int, mode: str = "quantum") -> None:
        if mode!= "quantum":
            raise ValueError("Quantum mode is mandatory for this implementation.")
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        """Construct a circuit that mirrors the classical
        self‑attention shape.
        """
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)

        # Rotation stage – one rotation per qubit
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], qr[i])
            qc.ry(rotation_params[3 * i + 1], qr[i])
            qc.rz(rotation_params[3 * i + 2], qr[i])

        # Entanglement stage – a chain of controlled‑RX gates
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], qr[i], qr[i + 1])

        qc.measure(qr, cr)
        return qc

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """Execute the attention circuit and return a probability
        distribution that can be interpreted as attention weights.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape ``(3 * n_qubits,)`` encoding the single‑qubit
            rotations.
        entangle_params : np.ndarray
            Array of shape ``(n_qubits - 1,)`` encoding the controlled‑RX
            angles.
        shots : int, default 1024
            Number of shots for the simulation.

        Returns
        -------
        np.ndarray
            Normalised probability vector over the ``2**n_qubits`` basis
            states.  The vector is reshaped to ``(n_qubits, 2)`` so that
            each qubit contributes a binary attention weight.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, self.backend, shots=shots)
        result = job.result().get_counts(qc)

        # Convert counts to a probability vector
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, count in result.items():
            idx = int(bitstring[::-1], 2)  # reverse due to Qiskit ordering
            probs[idx] = count / shots

        # Reshape to per‑qubit binary probabilities
        return probs.reshape(self.n_qubits, 2)

    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int):
        """Quantum dataset generator identical to the ML seed but
        returning complex amplitudes for consistency with the quantum
        regression example.
        """
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels

__all__ = ["HybridSelfAttention"]
