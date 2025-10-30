"""Quantum helper for the hybrid autoencoder.

Provides a parameterised variational circuit combined with a swap‑test
that yields a single expectation value per latent vector.  The circuit is
passed to a Qiskit :class:`SamplerQNN` wrapper so it can be used as a PyTorch
module.  The wrapper treats the latent vector as weight parameters and
evaluates the circuit with Qiskit’s ``StatevectorSampler``.
"""
from __future__ import annotations

from typing import Iterable

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

import torch
from torch import nn


def get_quantum_latent_circuit(num_qubits: int) -> QuantumCircuit:
    """Return a RealAmplitudes ansatz followed by a swap‑test.

    The circuit has ``num_qubits`` data qubits and one auxiliary qubit that
    is measured to produce the latent representation.  It is used as the
    weight‑parameterised part of a :class:`SamplerQNN`.
    """
    qr = QuantumRegister(num_qubits + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Variational ansatz on the data qubits
    circuit.compose(RealAmplitudes(num_qubits, reps=3), range(0, num_qubits), inplace=True)

    # Swap‑test with an auxiliary qubit
    aux = num_qubits
    circuit.h(aux)
    for i in range(num_qubits):
        circuit.cswap(aux, i, i + num_qubits)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit


class QuantumLatentLayer(SamplerQNN):
    """Wrapper around Qiskit’s ``SamplerQNN`` that accepts a latent vector as
    weight parameters and returns a scalar output per sample.

    The ``forward`` method converts a batch of latent vectors into a list of
    parameter dictionaries for the sampler, runs the circuit, and extracts the
    probability of measuring ``0`` on the auxiliary qubit.
    """

    def __init__(self, circuit: QuantumCircuit, latent_dim: int):
        # No classical input parameters; all trainable parameters are the latent vector
        super().__init__(
            circuit=circuit,
            input_params=[],
            weight_params=[f"w{i}" for i in range(latent_dim)],
            sampler=Sampler(),
            interpret=lambda x: x,
            output_shape=(1,),
        )
        self.latent_dim = latent_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum circuit for each latent vector in ``latent``.

        Parameters
        ----------
        latent : torch.Tensor
            Tensor of shape (batch, latent_dim) whose rows are interpreted as
            weight parameters for the variational circuit.

        Returns
        -------
        torch.Tensor
            Shape (batch,) containing the probability of measuring ``0`` on the
            auxiliary qubit for each sample.
        """
        batch = latent.shape[0]
        param_list: list[dict[str, float]] = []

        # Build a list of parameter dictionaries for the sampler
        for i in range(batch):
            params = {f"w{j}": latent[i, j].item() for j in range(self.latent_dim)}
            param_list.append(params)

        # Run the sampler
        results = self.sampler.run(self.circuit, param_list)

        # Extract probability of measuring '0' on the auxiliary qubit
        probs = []
        for sv in results:
            probs.append(sv.probability("0", [self.circuit.num_qubits - 1]))
        return torch.tensor(probs, dtype=torch.float32, device=latent.device)


__all__ = ["get_quantum_latent_circuit", "QuantumLatentLayer"]
