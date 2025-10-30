"""Quantum‑circuit implementation of a self‑attention block with feature‑mapping.

The class `SelfAttentionGen` builds a parameterised Qiskit circuit that mirrors the
classical API.  It uses a RealAmplitudes ansatz for feature embedding and a
sequence of controlled‑RX gates to realize the “entanglement” part of the
attention mechanism.  The circuit is compiled for qasm simulation but is
backend‑agnostic.

Key design points
  * `rotation_params` – 3 angles per qubit for RX/RY/RZ rotations.
  * `entangle_params` – one parameter per CX‑like gate between consecutive qubits.
  * `run` returns a dictionary of measurement counts (or statevector if requested).
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def _build_ansatz(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """Feature‑embedding ansatz."""
    return RealAmplitudes(num_qubits, reps=reps)


class SelfAttentionGen:
    """Quantum self‑attention block."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)

        # Feature embedding – RX/RY/RZ per qubit
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)

        # Entangling block – controlled‑RX between neighbours
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rx(entangle_params[i], i + 1)

        # Optional measurement
        qc.measure_all()
        return qc

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
        statevector: bool = False,
    ):
        """
        Execute the attention circuit.

        Parameters
        ----------
        backend : qiskit backend (AER, IBM, etc.)
        rotation_params : (n_qubits*3,) array
            RX/RY/RZ angles per qubit.
        entangle_params : (n_qubits-1,) array
            RX angles for entangling gates.
        shots : int
            Number of repetitions for measurement.
        statevector : bool
            If True return a Statevector rather than measurement counts.

        Returns
        -------
        counts / statevector
        """
        qc = self._build_circuit(rotation_params, entangle_params)
        if statevector:
            return qiskit.execute(qc, backend).result().get_statevector(qc)
        job = qiskit.execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)


__all__ = ["SelfAttentionGen"]
