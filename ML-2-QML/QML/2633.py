import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def AutoencoderHybridQNN(
    latent_dim: int = 32,
    num_trash: int = 2,
    reps: int = 3,
) -> SamplerQNN:
    """
    Construct a hybrid quantum autoencoder.

    The circuit first encodes the input into ``latent_dim`` qubits via a
    RealAmplitudes ansatz.  A domain‑wall sub‑circuit is inserted
    to provide a non‑trivial initial state.  A swap‑test style
    measurement on an auxiliary qubit yields a 2‑dimensional output
    vector that is interpreted by the SamplerQNN.
    """
    algorithm_globals.random_seed = 42

    # Build the encoder part
    def encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz on latent + trash qubits
        qc.compose(
            RealAmplitudes(num_latent + num_trash, reps=reps),
            range(num_latent + num_trash),
            inplace=True,
        )
        qc.barrier()

        # Auxiliary qubit for swap test
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    qc = encoder_circuit(latent_dim, num_trash)

    # Domain wall sub‑circuit to inject structure
    def domain_wall(qc: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
        for i in range(start, end):
            qc.x(i)
        return qc

    # Insert domain wall on the trash qubits
    qc = domain_wall(qc, latent_dim, latent_dim + 2 * num_trash)

    # SamplerQNN wrapper
    sampler = StatevectorSampler()
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],  # no classical input parameters
        weight_params=qc.parameters,
        interpret=lambda x: x,  # identity interpret
        output_shape=2,
        sampler=sampler,
    )
    return qnn
