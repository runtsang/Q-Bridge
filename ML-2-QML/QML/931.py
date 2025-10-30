"""Quantum self‑attention using a parameterised circuit.

The implementation follows the original interface (`run(backend,
rotation_params, entangle_params, shots)`) but extends it with
multi‑head support and a variational ansatz.  Each head is realised as a
sub‑circuit that applies rotation gates followed by controlled‑X
entanglement.  The output is a probability distribution over the
attention keys, which can be used to weight classical value vectors.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers import Backend


class SelfAttention:
    """
    Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits per head.  The total qubits = n_heads * n_qubits.
    n_heads : int, default 1
        Number of attention heads.
    """

    def __init__(self, n_qubits: int = 4, n_heads: int = 1):
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.total_qubits = n_qubits * n_heads

    def _build_head_circuit(
        self,
        head_idx: int,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """
        Build a single‑head circuit.

        Parameters
        ----------
        head_idx : int
            Index of the head.
        rotation_params : np.ndarray
            Rotation angles of shape (n_qubits, 3) for RX, RY, RZ.
        entangle_params : np.ndarray
            Entanglement angles of shape (n_qubits - 1,).
        """
        qr = QuantumRegister(self.n_qubits, f"q{head_idx}")
        cr = ClassicalRegister(self.n_qubits, f"c{head_idx}")
        circuit = QuantumCircuit(qr, cr)

        # Apply rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[i, 0], qr[i])
            circuit.ry(rotation_params[i, 1], qr[i])
            circuit.rz(rotation_params[i, 2], qr[i])

        # Entangle neighbouring qubits with controlled‑X
        for i in range(self.n_qubits - 1):
            circuit.cx(qr[i], qr[i + 1])

        circuit.measure(qr, cr)
        return circuit

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """
        Assemble the full multi‑head circuit by concatenating sub‑circuits.
        """
        full_circuit = QuantumCircuit()
        for h in range(self.n_heads):
            head_rot = rotation_params[h]
            head_ent = entangle_params[h]
            sub_circ = self._build_head_circuit(h, head_rot, head_ent)
            full_circuit.append(sub_circ, full_circuit.qubits)
        return full_circuit

    def run(
        self,
        backend: Backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return a probability distribution over keys.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            Execution backend.
        rotation_params : np.ndarray
            Shape (n_heads, n_qubits, 3).
        entangle_params : np.ndarray
            Shape (n_heads, n_qubits - 1).
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Attention weights of shape (n_heads, n_qubits).
        """
        circ = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circ, backend=backend, shots=shots)
        result = job.result()

        # Convert counts to probabilities per head
        probs = np.zeros((self.n_heads, self.n_qubits))
        for h in range(self.n_heads):
            register_name = f"c{h}"
            counts = result.get_counts(circ, key=register_name)
            for outcome, cnt in counts.items():
                for i, bit in enumerate(reversed(outcome)):
                    probs[h, i] += cnt
            probs[h] /= shots
        return probs
