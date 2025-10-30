"""Quantum sampler network that mirrors the classical SamplerQNN.
The circuit uses a RealAmplitudes ansatz, a domain‑wall
preprocessing, and a swap‑test based autoencoder structure
from the Autoencoder.py reference. The output is sampled
by Qiskit\'s StatevectorSampler and interpreted as class
probabilities."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


def SamplerQNN(num_latent: int = 3, num_trash: int = 2, reps: int = 3):
    """
    Construct a quantum sampler that encodes a latent vector of size
    ``num_latent`` using a RealAmplitudes ansatz, then applies a
    swap test with ``num_trash`` ancillary qubits. The circuit
    is wrapped in a Qiskit ``SamplerQNN`` which outputs a
    probability distribution that can be interpreted as
    class probabilities.
    """
    algorithm_globals.random_seed = 42
    sampler = StatevectorSampler()

    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(
            RealAmplitudes(num_latent + num_trash, reps=reps),
            range(0, num_latent + num_trash),
            inplace=True,
        )
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def domain_wall(circuit: QuantumCircuit, a: int, b: int):
        for i in range(int(b / 2), int(b)):
            circuit.x(i)
        return circuit

    qc = auto_encoder_circuit(num_latent, num_trash)
    dw_circuit = domain_wall(
        QuantumCircuit(num_latent + 2 * num_trash), 0, num_latent + 2 * num_trash
    )
    qc.compose(dw_circuit, range(0, num_latent + 2 * num_trash), inplace=True)

    def interpret(probabilities):
        # The circuit outputs a single bit; we interpret the
        # probability of measuring |1> as the first class
        # and the remaining probability mass as the other classes.
        p1 = probabilities[1]
        return [p1, 1 - p1, 0.0, 0.0]

    sampler_qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=interpret,
        output_shape=4,
        sampler=sampler,
    )
    return sampler_qnn
