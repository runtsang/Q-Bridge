"""Quantum autoencoder with swap‑test latent comparison and domain‑wall injection."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def quantum_autoencoder(num_latent: int, num_trash: int) -> SamplerQNN:
    """
    Build a parameterized quantum circuit that encodes the input, applies a swap‑test
    against auxiliary trash qubits, and returns a SamplerQNN that can be used as a
    differentiable layer in a hybrid training loop.
    """
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    # Parameterised ansatz for the encoder block
    def ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=5)

    # Construct the full circuit
    def build_circuit() -> QuantumCircuit:
        total_qubits = num_latent + 2 * num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode the input into the first (num_latent + num_trash) qubits
        qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)

        # Swap‑test between latent and trash subspaces
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    qc = build_circuit()

    # Inject a domain‑wall pattern to break symmetry and enrich the state space
    def domain_wall(qc: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        for i in range(start, end):
            qc.x(i)
        return qc

    qc = domain_wall(qc, 0, num_latent + num_trash)

    # Interpret the measurement outcome as a probability vector
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

__all__ = ["quantum_autoencoder"]
