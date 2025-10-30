"""Quantum kernel implementation inspired by an autoencoder circuit."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler


class QuantumKernelMethod:
    """
    Quantum kernel that uses a variational autoencoder‑style circuit followed
    by a swap‑test.  The circuit is evaluated with a state‑vector sampler,
    making it compatible with both simulation and real hardware.
    """

    def __init__(
        self,
        latent_dim: int = 3,
        trash_dim: int = 2,
        repetitions: int = 5,
    ) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.repetitions = repetitions
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # raw probability amplitude
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a variational autoencoder with a swap‑test."""
        total_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encoder ansatz
        encoder = RealAmplitudes(
            num_qubits=self.latent_dim + self.trash_dim,
            reps=self.repetitions,
        )
        qc.compose(encoder, range(0, self.latent_dim + self.trash_dim), inplace=True)

        # Swap‑test
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two feature vectors by evaluating
        the sampler QNN.  The inputs are ignored because the circuit has no
        trainable input parameters; the kernel is purely based on the
        internal variational parameters.
        """
        # The QNN expects a batch of zero‑dimensional inputs; we provide a
        # placeholder tensor of shape (1,)
        dummy_in = torch.zeros((1,), dtype=torch.float32)
        k = self.qnn(dummy_in)
        # The swap‑test yields a 2‑element probability vector; we map it
        # to a kernel value by taking the probability of measuring |0>.
        return torch.tensor(k[0, 0].item(), dtype=torch.float32)

    def kernel_matrix(
        self,
        a: Sequence[Iterable[float]],
        b: Sequence[Iterable[float]],
    ) -> np.ndarray:
        """
        Evaluate the kernel Gram matrix between two datasets ``a`` and ``b``.
        Each element is a sequence of floats representing a feature vector.
        """
        # Convert inputs to tensors; the QNN ignores them, but we keep the
        # interface consistent with the classical counterpart.
        _ = [torch.as_tensor(v, dtype=torch.float32) for v in a + b]
        return np.array(
            [[self.forward(torch.tensor(x), torch.tensor(y)).item() for y in b] for x in a]
        )


__all__ = ["QuantumKernelMethod"]
