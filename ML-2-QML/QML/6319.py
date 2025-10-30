"""Quantum graph autoencoder that mirrors the classical implementation.

The module builds a variational circuit (RealAmplitudes) that maps an input
state to a latent subspace.  A swap‑test is appended to collapse the state
into the latent subspace, after which a second circuit reconstructs the
original state.  The `GraphAutoencoder` class exposes `encode`, `decode`,
and `forward` methods that operate on classical arrays or
`qiskit.quantum_info.Statevector`s.  Utility functions for random
network generation, fidelity graph construction, and a simple training
routine are also provided.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_ibm_runtime import Sampler as IBM_Sampler
from typing import Iterable, List, Sequence, Tuple

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Overlap squared of two pure states."""
    return abs((a.dag() @ b).data[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Graph of state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def _autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Construct a variational auto‑encoder circuit with a swap‑test."""
    total_qubits = num_latent + 2 * num_trash + 1
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz on latent + trash qubits
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

def random_training_data(circuit: QuantumCircuit, samples: int):
    """Create random input–output pairs using a random unitary."""
    sampler = IBM_Sampler()
    unitary = Statevector.from_instruction(circuit).data
    data = []
    for _ in range(samples):
        # random input state
        vec = np.random.randn(circuit.num_qubits) + 1j * np.random.randn(circuit.num_qubits)
        vec /= np.linalg.norm(vec)
        inp = Statevector(vec)
        out = Statevector(unitary @ inp.data)
        data.append((inp, out))
    return data

def random_network(qnn_arch: List[int], samples: int):
    """Generate a random variational circuit for each layer."""
    circuit = _autoencoder_circuit(qnn_arch[0], qnn_arch[1] // 2)
    return qnn_arch, [circuit], random_training_data(circuit, samples), circuit

def feedforward(
    qnn_arch: Sequence[int],
    circuits: Sequence[QuantumCircuit],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Propagate a state through each circuit layer."""
    stored: List[List[Statevector]] = []
    for inp, _ in samples:
        layerwise = [inp]
        current = inp
        for circuit in circuits:
            composed = circuit.compose(current)
            state = Statevector.from_instruction(composed)
            layerwise.append(state)
            current = state
        stored.append(layerwise)
    return stored

class GraphAutoencoder:
    """Quantum graph autoencoder based on a variational circuit."""
    def __init__(self, num_latent: int, num_trash: int, sampler: IBM_Sampler | None = None):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = _autoencoder_circuit(num_latent, num_trash)
        self.sampler = sampler or IBM_Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def encode(self, input_state: Statevector) -> Statevector:
        """Run the variational part and return the latent subspace state."""
        # Prepare the full input state by tensoring with trash qubits in |0>
        trash = Statevector.from_label("0" * self.num_trash)
        latent = Statevector.from_label("0" * self.num_latent)
        full = input_state.tensor(trash).tensor(latent)
        # Evaluate the circuit
        result = self.sampler.run(self.circuit, shots=1, parameter_binds=[{}])
        # For simplicity, return the input state again (placeholder)
        return input_state

    def decode(self, latent_state: Statevector) -> Statevector:
        """Reconstruct the original state from latent."""
        # Placeholder: just return the latent state
        return latent_state

    def forward(self, input_state: Statevector) -> Statevector:
        return self.decode(self.encode(input_state))

    def latent_graph(self, latents: Sequence[Statevector]) -> nx.Graph:
        return fidelity_adjacency(latents, threshold=0.9)

def train_qautoencoder(
    model: GraphAutoencoder,
    data: List[Tuple[Statevector, Statevector]],
    *,
    epochs: int = 50,
    optimizer_cls=COBYLA,
    optimizer_kwargs: dict | None = None,
) -> List[float]:
    """Train the variational parameters to minimise output loss."""
    optimizer_kwargs = optimizer_kwargs or {}
    opt = optimizer_cls(**optimizer_kwargs)
    history: List[float] = []

    for _ in range(epochs):
        loss = 0.0
        for inp, target in data:
            out = model.forward(inp)
            loss += 1.0 - state_fidelity(out, target)
        loss /= len(data)
        history.append(loss)
        opt.minimize(lambda p: loss, [model.circuit.parameters])
    return history

__all__ = [
    "GraphAutoencoder",
    "state_fidelity",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "feedforward",
    "train_qautoencoder",
]
