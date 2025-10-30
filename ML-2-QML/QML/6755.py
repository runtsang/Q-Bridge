"""Quantum helper for the hybrid autoencoder.

The module builds a variational circuit that implements a
swap‑test‑based latent representation, mirroring the original
`Autoencoder.py` QML seed.  It exposes a single factory
`create_qnn(num_latent, num_trash)` that returns a
`SamplerQNN` ready to be used as a layer in the classical network.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def _swap_test_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a circuit that performs a swap‑test on `num_latent` and
    `num_trash` qubits and measures an auxiliary qubit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz on the latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    qc.barrier()

    # Auxiliary qubit for the swap‑test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)

    qc.measure(aux, cr[0])
    return qc


def create_qnn(num_latent: int, num_trash: int) -> SamplerQNN:
    """Build a SamplerQNN that maps a latent vector to a 2‑qubit
    expectation value.  The returned QNN can be called like a
    torch `nn.Module` and is differentiable via the sampler."""
    # The circuit depends on the number of latent parameters.
    circuit = _swap_test_circuit(num_latent, num_trash)

    # Use the Aer QASM simulator for sampling
    sampler = Sampler()
    # No input parameters – all parameters are in the ansatz
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,  # raw output
        output_shape=2,
        sampler=sampler,
    )
    return qnn


# For convenience, expose a function that matches the original
# Autoencoder() signature from the QML seed.
def Autoencoder() -> SamplerQNN:
    """Return a default QNN configured with 3 latent qubits and 2 trash."""
    return create_qnn(num_latent=3, num_trash=2)
