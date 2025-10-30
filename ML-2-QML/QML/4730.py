"""Quantum implementation of :class:`AutoencoderHybrid`.

The class builds a parameterized quantum circuit that performs an
auto‑encoding operation: an input feature vector is encoded into a
quantum state, processed by a RealAmplitudes ansatz, and a swap‑test
domain wall extracts a latent vector.  The latent is produced by a
:class:`qiskit_machine_learning.neural_networks.SamplerQNN`, making the
outputs differentiable for hybrid training.

The API mirrors the classical implementation so that the same code can
select either backend.
"""
from __future__ import annotations

from typing import List, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli


def _build_ansatz(num_qubits: int, reps: int = 5) -> QuantumCircuit:
    """Return a RealAmplitudes ansatz circuit."""
    return RealAmplitudes(num_qubits, reps=reps)


def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Insert a domain wall (X gates) into the circuit."""
    for i in range(start, end):
        circuit.x(i)
    return circuit


def _autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Construct the full auto‑encoding circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz on latent + first trash block
    qc.compose(_build_ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test domain wall
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc


class AutoencoderHybrid(nn.Module):
    """Quantum auto‑encoder mirroring the classical API."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        trash_dim: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim

        # Build the underlying circuit.
        self.circuit = _autoencoder_circuit(latent_dim, trash_dim)

        # SamplerQNN to produce a differentiable latent vector.
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],           # No classical inputs – we use the statevector directly.
            weight_params=self.circuit.parameters,
            sampler=self.sampler,
            interpret=lambda x: x,     # Identity interpretation.
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of classical inputs into quantum latent vectors."""
        # Convert inputs to state vectors: here we simply embed each feature
        # as a rotation angle on the first few qubits for illustration.
        batch = x.detach().cpu().numpy()
        batch_size = batch.shape[0]
        latent_vectors = []
        for vec in batch:
            qc = self.circuit.copy()
            # Map input features to rotation angles on the first qubits.
            for i, angle in enumerate(vec[: self.latent_dim + self.trash_dim]):
                qc.ry(angle, i)
            # Evaluate the SamplerQNN to obtain a latent vector.
            output = self.qnn(qc)
            latent_vectors.append(output.detach().cpu().numpy())
        return torch.tensor(latent_vectors, device=self.device, dtype=torch.float32)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode quantum latent vectors back to classical space."""
        # In this toy example we simply apply a linear map.
        # A real implementation would involve a reverse quantum circuit.
        linear = nn.Linear(self.latent_dim, self.input_dim, bias=False).to(self.device)
        return linear(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(x)
        return self.decode(z)

    def evaluate(
        self,
        observables: Iterable[Pauli],
        parameter_sets: Iterable[Tuple[float,...]],
    ) -> List[List[complex]]:
        """Use a lightweight fast estimator to compute expectation values."""
        from qiskit.quantum_info import Statevector
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circuit = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)))
            state = Statevector.from_instruction(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = ["AutoencoderHybrid"]
