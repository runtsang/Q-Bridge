"""Quantum encoder for the hybrid autoencoder.

The module exposes a function Autoencoder that builds a variational
circuit with RealAmplitudes and a swap‑test based latent extraction.
It returns a Qiskit SamplerQNN that is a torch.nn.Module, so it can
be plugged into the classical decoder.  A helper ``latent_fidelity_graph``
builds a state‑fidelity graph over a set of input samples.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def Autoencoder(
    num_latent: int,
    num_trash: int = 2,
    reps: int = 5,
    seed: int = 42,
) -> SamplerQNN:
    """Return a variational encoder as a SamplerQNN.

    The circuit encodes the input into ``num_latent`` qubits and
    uses a swap‑test with ``num_trash`` ancilla qubits to produce
    a latent vector that can be interpreted as a probability
    distribution over the latent basis.
    """
    algorithm_globals.random_seed = seed
    sampler = StatevectorSampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=reps)

    def encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode the data into the first ``num_latent`` qubits
        qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)

        # Swap‑test with ancilla
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    circuit = encoder_circuit(num_latent, num_trash)

    # The circuit has no explicit input parameters; we treat the
    # first ``num_latent`` qubits as the data register and the rest
    # as junk.  The sampler will produce the probability of the
    # measured ancilla being |1>, which we interpret as the latent
    # value for that qubit.
    interpret = lambda x: x  # Identity – the sampler already returns a vector

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


def latent_fidelity(state_a, state_b):
    """Return the squared overlap between two statevectors."""
    return abs((state_a.dag() * state_b)[0, 0]) ** 2


def latent_fidelity_graph(
    circuit: QuantumCircuit,
    inputs: np.ndarray,
    threshold: float = 0.8,
    secondary: float | None = None,
) -> nx.Graph:
    """Build a graph of latent states for a set of input samples."""
    sampler = StatevectorSampler()
    graph = nx.Graph()
    states = []

    for inp in inputs:
        # Prepare a statevector for the input
        sv = sampler.run(circuit, shots=1, parameter_binds=inp).result().get_statevector()
        states.append(sv)

    graph.add_nodes_from(range(len(states)))
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fid = latent_fidelity(states[i], states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary)
    return graph


__all__ = ["Autoencoder", "latent_fidelity_graph"]
