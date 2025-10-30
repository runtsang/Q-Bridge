"""Quantum sampler integrating attention entanglement and autoencoder ansatz."""

from __future__ import annotations

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.circuit.library import RealAmplitudes


class SamplerQNNQuantum:
    """Variational circuit that receives classical attention parameters
    and performs a swap‑test based autoencoder sampling."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_trash: int = 2,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash

        # Attention parameters
        self.rot_params = ParameterVector("rot", 3 * input_dim)
        self.ent_params = ParameterVector("ent", input_dim - 1)

        # Autoencoder ansatz parameters
        self.aa_params = ParameterVector(
            "aa", 5 * (latent_dim + num_trash) * 5
        )  # placeholder size

        # Build the circuit
        self.circuit = self._build_circuit()

        # Sampler primitive
        self.sampler = StatevectorSampler()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Autoencoder ansatz
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=5)
        qc.compose(ansatz, range(0, self.latent_dim + self.num_trash), inplace=True)
        qc.barrier()

        # Domain wall (simple X‑gates on trash qubits)
        for i in range(self.num_trash, self.latent_dim + self.num_trash):
            qc.x(i)

        # Swap test with auxiliary qubit
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)

        qc.measure(aux, cr[0])
        return qc

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        # Map inputs to rotation parameters
        rot_vals = inputs.detach().cpu().numpy().flatten()
        # Entanglement parameters set to zero for simplicity
        ent_vals = np.zeros(self.ent_params.size)

        # Bind parameters
        bound = self.circuit.bind_parameters(
            {p: v for p, v in zip(self.rot_params, rot_vals)}
        )

        # Execute sampler
        result = self.sampler.run(bound).result()
        counts = result.get_counts(bound)
        total = sum(counts.values())
        probs = np.array([counts.get("0", 0), counts.get("1", 0)]) / total
        return torch.as_tensor(probs, dtype=inputs.dtype, device=inputs.device)


__all__ = ["SamplerQNNQuantum"]
