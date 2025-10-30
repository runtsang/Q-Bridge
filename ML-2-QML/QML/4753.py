"""Quantum module that builds a swap‑test based SamplerQNN for the hybrid autoencoder.

The circuit encodes the latent vector as parameters of a RealAmplitudes ansatz
on a register of `quantum_latent_dim` qubits.  A controlled‑SWAP with a
fixed reference register implements the swap‑test; measuring the ancilla
gives a fidelity estimate between the encoded state and |0…0⟩.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA


def _build_swap_test_circuit(
    quantum_latent_dim: int,
    quantum_reps: int,
) -> QuantumCircuit:
    """Return a swap‑test circuit with a parameterized ansatz.

    Parameters
    ----------
    quantum_latent_dim : int
        Number of qubits that encode the latent vector.
    quantum_reps : int
        Repetitions of the RealAmplitudes ansatz.

    The circuit has 1 + 2 * quantum_latent_dim qubits:
    - qubit 0 : ancilla
    - qubits 1..quantum_latent_dim : register A (parameterised)
    - qubits 1+quantum_latent_dim.. : register B (fixed |0…0⟩)
    """
    ancilla = 0
    regA_start = 1
    regB_start = regA_start + quantum_latent_dim
    total_qubits = 1 + 2 * quantum_latent_dim

    qc = QuantumCircuit(total_qubits)

    # Prepare ancilla in |+⟩
    qc.h(ancilla)

    # Prepare reference register B in |0…0⟩ – no operation needed

    # Parameterised ansatz on register A
    theta = ParameterVector("theta", quantum_latent_dim * quantum_reps)
    ansatz = RealAmplitudes(quantum_latent_dim, reps=quantum_reps)
    ansatz = ansatz.assign_parameters(theta)
    qc.compose(ansatz, range(regA_start, regA_start + quantum_latent_dim), inplace=True)

    # Controlled‑SWAP between A and B, controlled by ancilla
    for i in range(quantum_latent_dim):
        qc.cswap(ancilla, regA_start + i, regB_start + i)

    # Measure ancilla
    qc.h(ancilla)
    qc.measure(ancilla, 0)

    return qc


def _fidelity_interpret(x: np.ndarray) -> float:
    """Convert the ancilla measurement probability to a fidelity estimate.

    P(0) = (1 + F)/2   →   F = 2*P(0) - 1
    """
    return float(2 * x[0] - 1)


def QuantumAutoencoderQNN(
    quantum_latent_dim: int,
    quantum_reps: int,
) -> SamplerQNN:
    """Return a SamplerQNN that implements the swap‑test circuit.

    The QNN has `quantum_latent_dim * quantum_reps` trainable parameters
    that will be supplied by the latent vector during training.
    """
    qc = _build_swap_test_circuit(quantum_latent_dim, quantum_reps)
    sampler = Aer.get_backend("qasm_simulator")
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=[p for p in qc.parameters],
        interpret=_fidelity_interpret,
        output_shape=1,
        sampler=sampler,
        optimizer=COBYLA(maxiter=50),
    )
    return qnn
