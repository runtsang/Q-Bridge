"""Quantum self‑attention using a variational circuit.

The circuit encodes per‑qubit rotations and pairwise entanglement.
After measurement it produces a probability distribution over
attention heads.  The run method returns a dictionary mapping
head index to probability of the qubit being in state |1>.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.providers import Backend

class SelfAttention:
    """
    Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits (default 4).  Each qubit represents an
        attention head.
    """

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend: Backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict[int, float]:
        """
        Execute the circuit and return a probability distribution
        over attention heads.

        Parameters
        ----------
        backend : qiskit.providers.Backend
            The Qiskit backend to run the circuit on.
        rotation_params : np.ndarray
            Shape (3 * n_qubits,) – rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits - 1,) – angles for CRX gates.
        shots : int, optional
            Number of shots for the simulation (default 1024).

        Returns
        -------
        dict[int, float]
            Mapping from head index to probability of the qubit
            being measured in state |1>.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        # Compute probabilities per head
        probs = np.zeros(self.n_qubits)
        total = sum(counts.values())
        for bitstring, cnt in counts.items():
            # bitstring is in little‑endian order: bitstring[0] is qubit 0
            for i, bit in enumerate(reversed(bitstring)):
                probs[i] += cnt * int(bit)
        probs /= total
        return {i: float(probs[i]) for i in range(self.n_qubits)}

__all__ = ["SelfAttention"]
