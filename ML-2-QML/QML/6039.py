"""Quantum counterpart of SelfAttentionHybrid.

The :class:`QuantumSelfAttentionHybrid` class builds two parameterised
circuits: a self‑attention style block and a quantum auto‑encoder based on
swap‑test and a RealAmplitudes ansatz.  The two blocks are composed into a
single circuit that can be executed on a Qiskit simulator or real device.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector

class QuantumSelfAttentionHybrid:
    """Quantum implementation of a self‑attention + auto‑encoder hybrid."""
    def __init__(self, n_qubits: int, latent: int, trash: int):
        self.n_qubits = n_qubits
        self.latent = latent
        self.trash = trash
        self.backend = Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(self.n_qubits, "q")
        self.cr = ClassicalRegister(self.n_qubits, "c")
        self.attn_circ = self._build_attention_circuit()
        self.auto_circ = self._build_autoencoder_circuit()
        self.combined = self._compose_circuits()

    def _build_attention_circuit(self) -> QuantumCircuit:
        """Builds a parameterised self‑attention style circuit."""
        circ = QuantumCircuit(self.qr, self.cr)
        # Parameterised rotations
        for i in range(self.n_qubits):
            circ.rx(np.pi/4, i)
            circ.ry(np.pi/6, i)
            circ.rz(np.pi/8, i)
        # Simplified entangling layer
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def _build_autoencoder_circuit(self) -> QuantumCircuit:
        """Builds a quantum auto‑encoder using a swap‑test style."""
        qr = QuantumRegister(self.latent + 2 * self.trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(self.latent + self.trash, reps=5)
        circ.compose(ansatz, slice(0, self.latent + self.trash), inplace=True)
        circ.barrier()
        aux = self.latent + 2 * self.trash
        circ.h(aux)
        for i in range(self.trash):
            circ.cswap(aux, self.latent + i, self.latent + self.trash + i)
        circ.h(aux)
        circ.measure(aux, cr[0])
        return circ

    def _compose_circuits(self) -> QuantumCircuit:
        """Combine attention and auto‑encoder circuits."""
        combined = QuantumCircuit(self.qr, self.cr)
        combined.append(self.attn_circ, self.qr)
        combined.append(self.auto_circ, self.qr)
        return combined

    def run(self, shots: int = 1024) -> dict:
        """Execute the combined circuit and return measurement counts."""
        job = execute(self.combined, self.backend, shots=shots)
        return job.result().get_counts(self.combined)

    def statevector(self) -> Statevector:
        """Return the statevector of the combined circuit (simulation only)."""
        simulator = Aer.get_backend("statevector_simulator")
        job = execute(self.combined, simulator)
        return Statevector.from_dict(job.result().get_statevector(self.combined))

__all__ = ["QuantumSelfAttentionHybrid"]
