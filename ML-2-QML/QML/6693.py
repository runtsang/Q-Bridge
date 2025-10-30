"""Quantum helper for hybrid autoencoder.

Provides a swap‑test based quantum autoencoder circuit and a convenience
function that evaluates the circuit to refine a latent vector. The
circuit encodes the latent vector into the angles of a RealAmplitudes
ansatz and uses a single auxiliary qubit to perform a swap test
between a reference trash state and the latent state. The measurement
probability of the auxiliary qubit is used as a refinement factor.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector


def quantum_autoencoder_circuit(
    num_latent: int,
    num_trash: int = 2,
    reps: int = 5,
) -> QuantumCircuit:
    """Build a swap‑test based quantum autoencoder circuit.

    Args:
        num_latent: Number of latent qubits.
        num_trash: Number of auxiliary qubits used for the swap test.
        reps: Number of repetitions of the RealAmplitudes ansatz.

    Returns:
        A :class:`~qiskit.circuit.QuantumCircuit` that accepts the latent
        vector as input parameters and performs a swap test against a
        reference trash state.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Encode latent and trash qubits with a RealAmplitudes ansatz
    circuit.compose(
        RealAmplitudes(num_latent + num_trash, reps=reps),
        range(0, num_latent + num_trash),
        inplace=True,
    )
    circuit.barrier()

    # Auxiliary qubit for swap test
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)

    # Measure auxiliary qubit
    circuit.measure(aux, cr[0])
    return circuit


def quantum_refine(
    latent: np.ndarray,
    params: np.ndarray,
    num_trash: int = 2,
    reps: int = 5,
) -> np.ndarray:
    """Evaluate the quantum autoencoder circuit to refine a latent vector.

    The function runs the circuit for each sample in ``latent`` using the
    provided ``params`` as the ansatz parameters. The probability of
    measuring ``0`` on the auxiliary qubit is used as a scaling factor
    applied to the latent vector, producing a refined latent vector.

    Args:
        latent: Latent vectors of shape ``(batch, latent_dim)``.
        params: Variational parameters for the RealAmplitudes ansatz.
        num_trash: Number of auxiliary qubits.
        reps: Number of ansatz repetitions.

    Returns:
        Refined latent vectors of the same shape as ``latent``.
    """
    num_latent = latent.shape[-1]
    circuit = quantum_autoencoder_circuit(num_latent, num_trash, reps)
    sampler = Sampler()

    # Prepare input parameters for the ansatz
    # The ansatz expects 2 * reps * (num_latent + num_trash) parameters
    num_ansatz_params = 2 * reps * (num_latent + num_trash)
    if latent.ndim == 2:
        # Pad each sample to the required number of parameters
        padding = np.zeros((latent.shape[0], num_ansatz_params - latent.shape[1]))
        input_params = np.concatenate([latent, padding], axis=1)
    else:
        padding = np.zeros(num_ansatz_params - latent.shape[0])
        input_params = np.concatenate([latent, padding])

    # Run the sampler for each sample
    job = sampler.run(circuit, input_params=input_params, weight_params=params)
    result = job.result()
    # The sampler returns a list of probability distributions for each sample
    probs = result.quasi_dists[0].data
    # Compute the probability of measuring 0 on the auxiliary qubit
    prob0 = np.array(
        [sum(counts.get("0", 0)) / sum(counts.values()) for counts in probs]
    )
    # Scale the latent vector by the probability
    refined = latent * prob0[:, None]
    return refined
