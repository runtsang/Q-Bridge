"""Quantum variational autoencoder helper using a RealAmplitudes ansatz and swap‑test similarity."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler as Sampler

def Autoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
    reps: int = 5,
    seed: int = 42,
) -> SamplerQNN:
    """Build a variational autoencoder circuit and wrap it as a QNN."""
    algorithm_globals.random_seed = seed
    sampler = Sampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        """Parameter‑shared RealAmplitudes ansatz."""
        return RealAmplitudes(num_qubits, reps=reps)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        """Full circuit with swap‑test for latent similarity."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # encode part
        circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()

        # swap‑test
        auxiliary = num_latent + 2 * num_trash
        circuit.h(auxiliary)
        for i in range(num_trash):
            circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary)
        circuit.measure(auxiliary, cr[0])

        return circuit

    circuit = auto_encoder_circuit(num_latent, num_trash)

    # Optional domain‑wall injection
    def domain_wall(circ: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
        for i in range(a, b):
            circ.x(i)
        return circ

    circuit = domain_wall(circuit, 0, num_latent + num_trash)

    # QNN wrapper
    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],  # no classical inputs
        weight_params=circuit.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn
