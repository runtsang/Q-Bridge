"""
AutoencoderGen230: Quantum autoencoder module.
"""

from __future__ import annotations

import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning import algorithm_globals

def QuantumAutoencoder(
    latent_dim: int = 3,
    num_trash: int = 2,
    reps: int = 5,
    seed: int = 42,
) -> SamplerQNN:
    algorithm_globals.random_seed = seed
    sampler = StatevectorSampler()
    def ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=reps)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        total_qubits = num_latent + 2 * num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.append(ansatz(num_latent + num_trash), list(range(num_latent + num_trash)))
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    circuit = auto_encoder_circuit(latent_dim, num_trash)

    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

__all__ = ["QuantumAutoencoder"]
