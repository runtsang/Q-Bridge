"""Quantum hybrid autoencoder leveraging a convolutional filter and a swap‑test latent layer."""

import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.random import random_circuit
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def _convolution_filter(num_qubits: int, threshold: float) -> QuantumCircuit:
    """Build a quantum filter that maps a classical kernel to a quantum state."""
    qc = QuantumCircuit(num_qubits)
    theta = [qc.params[i] for i in range(num_qubits)]
    for i in range(num_qubits):
        qc.rx(theta[i], i)
    qc.barrier()
    qc += random_circuit(num_qubits, 2)
    return qc

def HybridAutoencoder(kernel_size: int = 2,
                      latent_dim: int = 3,
                      shots: int = 1024,
                      threshold: float = 127) -> SamplerQNN:
    """Return a SamplerQNN that implements a convolution‑based quantum autoencoder."""
    # Determine qubit counts
    num_trash = kernel_size ** 2
    num_latent = latent_dim
    total_qubits = num_latent + 2 * num_trash + 1

    # Construct the quantum circuit
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # 1. Convolution filter on the first num_trash qubits
    conv = _convolution_filter(num_trash, threshold)
    qc.compose(conv, range(num_trash), inplace=True)

    # 2. Variational ansatz on latent + first trash register
    ansatz = RealAmplitudes(num_latent + num_trash, reps=3)
    qc.append(ansatz, range(num_latent, num_latent + num_trash))

    # 3. Swap‑test between latent and second trash register
    aux = total_qubits - 1
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    # Sampler and QNN
    sampler = StatevectorSampler()
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: torch.tensor(x, dtype=torch.float32),
        output_shape=latent_dim,
        sampler=sampler,
    )
    return qnn

__all__ = ["HybridAutoencoder"]
