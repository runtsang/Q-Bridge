"""
Quantum component of the hybrid autoencoder.  This module exposes a
SamplerQNN that implements the swap‑test‑based autoencoder circuit.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def QuantumAutoencoderCircuit(latent_dim: int, trash_dim: int, reps: int = 3) -> QuantumCircuit:
    """
    Build a variational quantum autoencoder circuit.

    Parameters
    ----------
    latent_dim : int
        Number of qubits that encode the latent vector.
    trash_dim : int
        Number of auxiliary qubits used for the swap test.
    reps : int
        Depth of the RealAmplitudes ansatz.

    Returns
    -------
    QuantumCircuit
        The fully‑constructed circuit ready to be wrapped by a SamplerQNN.
    """
    n_qubits = latent_dim + 2 * trash_dim + 1
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode data into the first block
    ansatz = RealAmplitudes(latent_dim + trash_dim, reps=reps)
    circuit.compose(ansatz, range(0, latent_dim + trash_dim), inplace=True)

    # Swap test
    auxiliary = latent_dim + 2 * trash_dim
    circuit.h(auxiliary)
    for i in range(trash_dim):
        circuit.cswap(auxiliary, latent_dim + i, latent_dim + trash_dim + i)
    circuit.h(auxiliary)

    circuit.measure(auxiliary, cr[0])
    return circuit


def QuantumAutoencoder(
    latent_dim: int,
    trash_dim: int | None = None,
    reps: int = 3,
) -> SamplerQNN:
    """
    Factory that returns a differentiable quantum autoencoder.

    Parameters
    ----------
    latent_dim : int
        Size of the latent space.
    trash_dim : int, optional
        Number of trash qubits; defaults to half of latent_dim.
    reps : int
        Depth of the variational ansatz.

    Returns
    -------
    SamplerQNN
        The quantum autoencoder ready for integration into a hybrid model.
    """
    if trash_dim is None:
        trash_dim = max(1, latent_dim // 2)

    circuit = QuantumAutoencoderCircuit(latent_dim, trash_dim, reps)

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,  # identity
        output_shape=(latent_dim,),
    )
    return qnn


__all__ = ["QuantumAutoencoderCircuit", "QuantumAutoencoder"]
