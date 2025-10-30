"""Quantum sampler component for SamplerQNN__gen294.

The quantum part mirrors the classical pipeline: the latent
representation produced by the encoder is fed as *weight* parameters
to a small variational circuit that outputs a probability
distribution over a single classical bit.  The circuit is
implemented with Qiskit and the :class:`qiskit_machine_learning.neural_networks.SamplerQNN`
wrapper for easy evaluation.
"""

from __future__ import annotations

import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class SamplerQNN__gen294:
    """Quantum sampler that accepts a latent vector as a set of
    rotation angles and returns a probability distribution.
    """
    def __init__(self, latent_dim: int, num_trash: int = 2) -> None:
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],  # no separate input parameters
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a simple variational circuit with a swap‑test
        that uses the latent vector as rotation angles."""
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent vector into first part of the register
        qc.compose(
            RealAmplitudes(self.latent_dim + self.num_trash, reps=5),
            range(self.latent_dim + self.num_trash),
            inplace=True,
        )
        qc.barrier()

        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def sample(self, latent: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum sampler for a batch of latent vectors."""
        # latent shape: (batch, latent_dim)
        probs = []
        for vec in latent:
            # convert torch tensor to list of floats
            vec_list = vec.tolist()
            out = self.qnn.forward(vec_list)
            probs.append(out)
        return torch.stack(probs)

__all__ = ["SamplerQNN__gen294"]
