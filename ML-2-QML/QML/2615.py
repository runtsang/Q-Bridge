"""Quantum component for the hybrid autoencoder.

Provides a SamplerQNN that implements a swap‑test based variational
circuit.  The circuit is built from a RealAmplitudes ansatz followed
by a swap‑test that compares the latent state to a fixed reference
state.  The output is the probability that the auxiliary qubit
measures 0, which is used as a fidelity proxy in the classical loss.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def build_autoencoder_qnn(
    latent_dim: int,
    num_trash: int = 2,
    reps: int = 5,
) -> SamplerQNN:
    """Return a differentiable QNN that maps a latent vector to a
    probability of the auxiliary qubit being 0.
    """
    # Parameter names for the latent vector
    latent_params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(latent_dim)]

    # Build the circuit
    qr = QuantumRegister(latent_dim + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode latent as Ry rotations
    for i, p in enumerate(latent_params):
        circuit.ry(p, qr[i])

    # Variational ansatz on the latent+trash qubits
    circuit.compose(
        RealAmplitudes(latent_dim + num_trash, reps=reps),
        range(0, latent_dim + num_trash),
        inplace=True,
    )

    # Swap‑test with auxiliary qubit
    circuit.h(0)
    for i in range(num_trash):
        circuit.cswap(0, latent_dim + i, latent_dim + num_trash + i)
    circuit.h(0)

    # Measurement on auxiliary qubit
    circuit.measure(0, cr[0])

    # Interpret the measurement as a probability for |0>
    def interpret(x: np.ndarray) -> np.ndarray:
        # x is a 2‑element array of probabilities for |0> and |1>
        return x[0]

    # Sampler for state‑vector simulation
    sampler = StatevectorSampler()

    # Separate input and weight parameters
    input_params = latent_params
    weight_params = [p for p in circuit.parameters if p not in input_params]

    return SamplerQNN(
        circuit=circuit,
        input_params=input_params,
        weight_params=weight_params,
        interpret=interpret,
        output_shape=1,
        sampler=sampler,
    )

__all__ = ["build_autoencoder_qnn"]
