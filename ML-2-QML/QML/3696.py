"""
AutoencoderGen343 – Quantum component

This module implements a quantum autoencoder that uses a RealAmplitudes
ansatz and a swap‑test to compare latent and original states.  A
parameter‑ised SamplerQNN is provided to read out the latent
representation as a probability distribution.  The circuit is
designed to be composable with the classical encoder from
``Autoencoder__gen343.py``.
"""

from __future__ import annotations

import numpy as np
import warnings

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Helper: domain‑wall insertion
# --------------------------------------------------------------------------- #
def domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """
    Flip qubits between ``start`` (inclusive) and ``end`` (exclusive) to
    inject domain‑wall noise for robustness tests.
    """
    for i in range(start, end):
        circuit.x(i)
    return circuit


# --------------------------------------------------------------------------- #
# Core quantum autoencoder circuit
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(
    num_latent: int,
    num_trash: int,
    reps: int = 5,
    domain_wall_start: int | None = None,
    domain_wall_end: int | None = None,
) -> QuantumCircuit:
    """
    Build a swap‑test based autoencoder.

    Parameters
    ----------
    num_latent : int
        Number of latent qubits.
    num_trash : int
        Number of auxiliary “trash” qubits used in the swap test.
    reps : int
        Repetitions of the RealAmplitudes ansatz.
    domain_wall_start / domain_wall_end : int | None
        Optional indices to apply a domain‑wall before the swap test.
    """
    total_qubits = num_latent + 2 * num_trash + 1
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Ansatz on latent + first trash block
    circuit.compose(
        RealAmplitudes(num_latent + num_trash, reps=reps), range(0, num_latent + num_trash), inplace=True
    )
    circuit.barrier()

    # Optional domain wall
    if domain_wall_start is not None and domain_wall_end is not None:
        circuit = domain_wall(circuit, domain_wall_start, domain_wall_end)

    # Swap test
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    return circuit


# --------------------------------------------------------------------------- #
# SamplerQNN wrapper (quantum version)
# --------------------------------------------------------------------------- #
def quantum_sampler_qnn(
    circuit: QuantumCircuit,
    input_params: list,
    weight_params: list,
    interpret: Callable[[np.ndarray], np.ndarray] | None = None,
    output_shape: int = 2,
) -> SamplerQNN:
    """
    Build a SamplerQNN around a given circuit.

    The function is deliberately lightweight so that it can be dropped
    into training loops or used as a drop‑in replacement for the classical
    SamplerQNN defined in ``Autoencoder__gen343.py``.
    """
    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=input_params,
        weight_params=weight_params,
        sampler=sampler,
        interpret=interpret,
        output_shape=output_shape,
    )
    return qnn


# --------------------------------------------------------------------------- #
# End‑to‑end helper to expose a comparable interface to the classical
# Autoencoder function.
# --------------------------------------------------------------------------- #
def Autoencoder() -> SamplerQNN:
    """
    Instantiate a quantum autoencoder sampler network.

    The returned object behaves like the classical SamplerQNN but
    internally runs a quantum circuit that implements a swap‑test
    autoencoder.  The output shape is 2 (probability of ancilla
    measurement 0/1) and can be interpreted as a similarity score
    between the latent and original states.
    """
    algorithm_globals.random_seed = 42
    num_latent, num_trash = 3, 2
    circuit = quantum_autoencoder_circuit(num_latent, num_trash)

    # Parameters for the circuit
    input_params = []  # no external inputs in this simple example
    weight_params = list(circuit.parameters)

    # Simple identity interpretation
    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    return quantum_sampler_qnn(
        circuit=circuit,
        input_params=input_params,
        weight_params=weight_params,
        interpret=identity_interpret,
        output_shape=2,
    )


__all__ = ["quantum_autoencoder_circuit", "quantum_sampler_qnn", "Autoencoder"]
