from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Swap‑test based quantum auto‑encoder circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode input data into first (num_latent+num_trash) qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test over trash qubits
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)

    qc.measure(aux, cr[0])
    return qc


def build_quantum_autoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
    seed: int = 42,
    backend=None,
    shots: int = 1024,
) -> SamplerQNN:
    """
    Returns a SamplerQNN that implements the swap‑test auto‑encoder.
    The QNN can be used as a kernel or generative block in hybrid workflows.
    """
    algorithm_globals.random_seed = seed
    sampler = Sampler(backend=backend, shots=shots)
    qc = quantum_autoencoder_circuit(num_latent, num_trash)

    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],          # No trainable input parameters
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=2,           # Measurement result of the auxiliary qubit
        sampler=sampler,
    )
    return qnn


__all__ = ["quantum_autoencoder_circuit", "build_quantum_autoencoder"]
