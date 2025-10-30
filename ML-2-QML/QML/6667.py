"""Quantum autoencoder using a variational circuit and statevector sampling."""

import numpy as np
import json
import time
import warnings

import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN


def Autoencoder():
    """Return a Qiskit SamplerQNN representing a quantum autoencoder."""
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        """Variational ansatz used for the latent part."""
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        """Construct the full auto‑encoding circuit with swap‑test."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode part
        circuit.compose(
            ansatz(num_latent + num_trash),
            range(0, num_latent + num_trash),
            inplace=True,
        )
        circuit.barrier()

        # Swap‑test
        auxiliary_qubit = num_latent + 2 * num_trash
        circuit.h(auxiliary_qubit)
        for i in range(num_trash):
            circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary_qubit)

        circuit.measure(auxiliary_qubit, cr[0])
        return circuit

    num_latent = 3
    num_trash = 2
    qc = auto_encoder_circuit(num_latent, num_trash)

    # Define interpret function that returns the probability of measuring 1
    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn
