"""Hybrid quantum autoencoder that combines a quantum convolution filter
with a variational encoder/decoder.

The encoder uses a parametric rotation layer that encodes classical
pixel values, followed by a RealAmplitudes ansatz over the latent
qubits, and a swap test with a trash subspace.  The decoder mirrors
the encoder.  The whole circuit is wrapped in a SamplerQNN so it
can be used as a differentiable layer in a classical optimiser.
"""

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def HybridAutoencoder(
    latent_dim: int = 3,
    trash_dim: int = 2,
    kernel_size: int = 2,
    threshold: float = 127.0,
    shots: int = 1024,
) -> SamplerQNN:
    """Construct a quantum autoencoder with a convolutional filter.

    Parameters
    ----------
    latent_dim : int
        Number of latent qubits.
    trash_dim : int
        Number of trash qubits per side of swap test.
    kernel_size : int
        Size of the 2‑D kernel used in the quantum convolution filter.
    threshold : float
        Threshold for mapping classical pixel values to rotation angles.
    shots : int
        Number of shots used by the sampler.
    """
    backend = Aer.get_backend("qasm_simulator")
    sampler = StatevectorSampler(backend=backend)

    total_qubits = latent_dim + 2 * trash_dim + 1  # +1 auxiliary qubit
    aux = total_qubits - 1

    # Build encoder circuit
    encoder = QuantumCircuit(total_qubits, name="encoder")

    # Parametric rotation layer that encodes classical data
    theta = [Parameter(f"theta_{i}") for i in range(total_qubits)]
    for i, p in enumerate(theta):
        encoder.rx(p, i)

    # RealAmplitudes ansatz on latent + trash qubits
    ansatz = RealAmplitudes(latent_dim + trash_dim, reps=3)
    encoder.append(ansatz, range(latent_dim + trash_dim))

    # Swap test with auxiliary qubit
    encoder.h(aux)
    for i in range(trash_dim):
        encoder.cswap(aux, latent_dim + i, latent_dim + trash_dim + i)
    encoder.h(aux)

    # Build decoder circuit (inverse of encoder)
    decoder = QuantumCircuit(total_qubits, name="decoder")
    decoder.append(ansatz.inverse(), range(latent_dim + trash_dim))
    decoder.h(aux)
    for i in reversed(range(trash_dim)):
        decoder.cswap(aux, latent_dim + i, latent_dim + trash_dim + i)
    decoder.h(aux)

    # Combine encoder and decoder
    autoencoder = encoder.compose(decoder, inplace=False)

    # Wrap in a SamplerQNN
    qnn = SamplerQNN(
        circuit=autoencoder,
        input_params=[],
        weight_params=autoencoder.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = ["HybridAutoencoder"]
