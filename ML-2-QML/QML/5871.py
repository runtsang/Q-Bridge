from __future__ import annotations

import numpy as np
from typing import Iterable

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def _swap_test(circuit: QuantumCircuit, aux: int, qubits: Iterable[int]) -> None:
    """Insert a swap‑test using ``aux`` as the auxiliary qubit."""
    circuit.h(aux)
    for q in qubits:
        circuit.cswap(aux, q, q + len(qubits))
    circuit.h(aux)

def build_autoencoder_circuit(num_latent: int, num_trash: int, reps: int = 5) -> QuantumCircuit:
    """
    Build the variational auto‑encoder circuit.

    Parameters
    ----------
    num_latent : int
        Number of latent qubits that encode the classical latent vector.
    num_trash : int
        Number of auxiliary qubits used in the swap‑test.
    reps : int, optional
        Number of repetitions of the RealAmplitudes ansatz.
    """
    total_qubits = num_latent + 2 * num_trash + 1
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode the latent part with a variational ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=reps)
    qc.compose(ansatz, list(range(num_latent + num_trash)), inplace=True)

    # Swap‑test with auxiliary qubit
    aux = num_latent + 2 * num_trash
    _swap_test(qc, aux, range(num_latent, num_latent + num_trash))

    # Measurement of the auxiliary qubit (used only for visualisation)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

def get_autoencoder_qnn(
    num_latent: int,
    num_trash: int,
    reps: int = 5,
    output_shape: int = 1,
) -> SamplerQNN:
    """
    Return a SamplerQNN that evaluates the auto‑encoder circuit.

    The QNN interprets the sampler output as the probability of
    measuring |1> on the auxiliary qubit.  This scalar can be fed
    back into the classical training loop.
    """
    qc = build_autoencoder_circuit(num_latent, num_trash, reps=reps)
    sampler = StatevectorSampler()

    def interpret(x: np.ndarray) -> np.ndarray:
        probs = np.abs(x) ** 2
        idx = 1 << (qc.num_qubits - 1)  # auxiliary qubit is the last qubit
        return np.array([probs[idx]])

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],            # no classical inputs
        weight_params=qc.parameters,
        interpret=interpret,
        output_shape=output_shape,
        sampler=sampler,
    )
    return qnn

__all__ = ["build_autoencoder_circuit", "get_autoencoder_qnn"]
