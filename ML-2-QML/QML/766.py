"""Quantum autoencoder with parameter‑efficient ansatz and hybrid loss."""
from __future__ import annotations

from typing import Callable

import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


class AutoencoderGen(SamplerQNN):
    """Hybrid quantum autoencoder with a lightweight ansatz and a fidelity‑based loss."""
    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        fidelity_weight: float = 0.5,
        sampler: Sampler | None = None,
    ) -> None:
        if sampler is None:
            sampler = Sampler()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.fidelity_weight = fidelity_weight
        circuit = self._build_circuit()
        super().__init__(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Constructs the swap‑test based autoencoder circuit."""
        # Domain‑wall preparation
        domain_wall = self._domain_wall_circuit(5, 0, 5)

        # Auto‑encoder ansatz
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=3)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap test
        qc.barrier()
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        # Merge domain‑wall and auto‑encoder
        qc.compose(domain_wall, range(self.num_latent + self.num_trash,
                                      self.num_latent + self.num_trash + 5), inplace=True)
        return qc

    @staticmethod
    def _domain_wall_circuit(num_qubits: int, a: int, b: int) -> QuantumCircuit:
        """Apply X gates to qubits in the interval [a, b)."""
        circ = QuantumCircuit(num_qubits)
        for i in range(a, b):
            circ.x(i)
        return circ

    def loss(self, params: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Hybrid loss: weighted sum of MSE and a cosine‑similarity proxy for fidelity."""
        mse = torch.mean((outputs - inputs) ** 2)
        # Cosine similarity as a proxy for state‑vector fidelity
        norm_inputs = torch.norm(inputs, dim=1, keepdim=True)
        norm_outputs = torch.norm(outputs, dim=1, keepdim=True)
        dot = torch.sum(inputs * outputs, dim=1, keepdim=True)
        cosine = dot / (norm_inputs * norm_outputs + 1e-8)
        fidelity = torch.mean(cosine)
        return (1 - self.fidelity_weight) * mse + self.fidelity_weight * (1 - fidelity)


__all__ = ["AutoencoderGen"]
