"""SelfAttentionGen404 – a parameterized quantum circuit emulating self‑attention."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import ParameterVector
from typing import Tuple, Dict


class SelfAttentionGen404:
    """
    Variational quantum circuit that produces a probability distribution over
    attention weights.  The circuit is parameterized by rotation and entanglement
    angles, mirroring the classical API:
        * ``run`` takes ``rotation_params`` and ``entangle_params`` and returns
          a dictionary mapping index pairs to probabilities.
    """

    def __init__(self, n_qubits: int, n_heads: int = 4):
        self.n_qubits = n_qubits
        self.n_heads = n_heads
        self.backend = AerSimulator()
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = None

    def _build_parameters(self,
                          rotation_params: np.ndarray,
                          entangle_params: np.ndarray) -> Tuple[ParameterVector, ParameterVector]:
        """
        Wrap raw numpy arrays into Qiskit parameters for circuit building.
        """
        rot = ParameterVector("rot", len(rotation_params))
        ent = ParameterVector("ent", len(entangle_params))
        return rot, ent

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        """
        Constructs a circuit where each qubit receives a rotation block,
        followed by a chain of controlled‑RX entanglements.
        """
        rot, ent = self._build_parameters(rotation_params, entangle_params)
        qc = QuantumCircuit(self.qr, self.cr)

        # Rotation layer
        for i in range(self.n_qubits):
            qc.rx(rot[3 * i], i)
            qc.ry(rot[3 * i + 1], i)
            qc.rz(rot[3 * i + 2], i)

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            qc.crx(ent[i], i, i + 1)

        # Measurement
        qc.measure(self.qr, self.cr)
        return qc

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> Dict[Tuple[int, int], float]:
        """
        Executes the circuit and returns a simplified attention matrix:
        each key is a pair (i, j) representing attention from qubit i to j,
        the value is the probability of measuring the state |ij⟩.
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        job = self.backend.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
        total = sum(counts.values())

        # Convert counts to a probability matrix
        attn_matrix: Dict[Tuple[int, int], float] = {}
        for bitstring, c in counts.items():
            # bitstring is in little‑endian order
            idx = tuple(int(bit) for bit in reversed(bitstring))
            if len(idx) == self.n_qubits:
                # pair the first two qubits as a simple (i, j) attention pair
                key = (idx[0], idx[1])
                attn_matrix[key] = attn_matrix.get(key, 0.0) + c / total

        return attn_matrix


__all__ = ["SelfAttentionGen404"]
