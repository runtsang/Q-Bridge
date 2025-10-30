"""Quantum sampler network with an autoencoder‑style ansatz.

The quantum implementation builds on the SamplerQNN helper from
qiskit‑machine‑learning.  A RealAmplitudes ansatz is used to
embed data, followed by a swap‑test based autoencoder that
captures a latent representation.  The circuit is sampled with
a StatevectorSampler, and the output is interpreted as a
probability distribution over two outcomes.

The public API mirrors the classical module: a class ``SamplerQNN``
and a factory function ``create_sampler_qnn`` that returns an
instance.  All imports are local to keep the module lightweight.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

__all__ = ["SamplerQNN", "create_sampler_qnn"]


class SamplerQNN:
    """Quantum sampler network that embeds an autoencoder ansatz.

    Parameters
    ----------
    latent_dim : int, optional
        Number of qubits used for the latent space.  Default is 3.
    trash_dim : int, optional
        Number of auxiliary qubits used in the swap‑test.  Default is 2.
    """

    def __init__(self, latent_dim: int = 3, trash_dim: int = 2) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the full quantum circuit.

        The circuit consists of:
        1. A RealAmplitudes feature embedding.
        2. A swap‑test based autoencoder that uses ``trash_dim`` auxiliary qubits.
        3. Measurement of a single ancilla qubit to produce a 2‑class output.
        """
        total_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Feature embedding (identity for simplicity)
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=5)
        qc.compose(ansatz, range(self.latent_dim + self.trash_dim), inplace=True)

        # Swap‑test autoencoder
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)

        # Measurement
        qc.measure(aux, cr[0])
        return qc

    def sampler(self) -> QSamplerQNN:
        """Return a qiskit‑machine‑learning SamplerQNN instance."""
        sampler = StatevectorSampler()
        return QSamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )


def create_sampler_qnn() -> SamplerQNN:
    """Factory that returns a ready‑to‑use quantum sampler network."""
    return SamplerQNN()
