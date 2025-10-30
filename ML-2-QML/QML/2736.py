"""Quantum implementation of a graph‑neural‑network autoencoder.

The quantum circuit mirrors the classical architecture: a stack of
variational layers followed by a swap‑test based autoencoder that
compresses the joint state into ``num_latent`` qubits.  The
functions below provide utilities for random network generation,
training data synthesis, forward propagation and fidelity‑based
graph construction.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator

Tensor = np.ndarray


def _random_qubit_state(num_qubits: int) -> Statevector:
    """Generate a random pure state on ``num_qubits``."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)


def random_training_data(target: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Create training pairs (input, target) for a quantum autoencoder."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        inp = _random_qubit_state(len(target.register))
        dataset.append((inp, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random target unitary and training data."""
    target = qk.quantum_info.RandomUnitary(2 ** qnn_arch[-1]).unitary
    target_sv = Statevector(target)
    training_data = random_training_data(target_sv, samples)
    return list(qnn_arch), training_data, target_sv


def feedforward(
    qnn_arch: Sequence[int],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Propagate each sample through a stack of RealAmplitudes layers."""
    states: List[List[Statevector]] = []
    for inp, _ in samples:
        layerwise = [inp]
        current = inp
        for layer in range(1, len(qnn_arch)):
            sub_circ = RealAmplitudes(qnn_arch[layer], reps=1)
            qc = QuantumCircuit(current.num_qubits)
            qc.append(sub_circ.to_instruction(), range(qnn_arch[layer]))
            current = Statevector.from_instruction(qc, initial_state=current)
            layerwise.append(current)
        states.append(layerwise)
    return states


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two pure states."""
    return abs(np.vdot(a.data, b.data)) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from quantum state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class GraphQNNAutoencoderQML:
    """
    Quantum counterpart of :class:`GraphQNNAutoencoder`.

    The circuit is built from a stack of ``RealAmplitudes`` layers
    followed by a variational autoencoder that compresses the joint
    state into ``num_latent`` qubits.  The autoencoder uses a
    swap‑test style measurement to enforce fidelity with the target
    unitary.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        num_latent: int,
        num_trash: int,
        sampler: AerSimulator | None = None,
    ) -> None:
        self.qnn_arch = list(qnn_arch)
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.sampler = sampler or AerSimulator()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode layer: RealAmplitudes on the first part of the register
        qc.append(
            RealAmplitudes(self.num_latent + self.num_trash, reps=5).to_instruction(),
            range(0, self.num_latent + self.num_trash),
        )

        # Swap‑test style autoencoder
        auxiliary = self.num_latent + 2 * self.num_trash
        qc.h(auxiliary)
        for i in range(self.num_trash):
            qc.cswap(auxiliary, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(auxiliary)
        qc.measure(auxiliary, cr[0])

        return qc

    def encode(self, state: Statevector) -> Statevector:
        """Apply the autoencoder circuit to a state and return the compressed part."""
        # Pad the state to match circuit size
        padded = Statevector.from_label("0" * (self.circuit.num_qubits - len(state.register)))
        padded = padded.tensor(state)
        result = padded.evolve(self.circuit)
        # Extract the first ``num_latent`` qubits
        return result.evolve(Statevector.from_label("0" * (self.circuit.num_qubits - self.num_latent)))

    def decode(self, latent: Statevector) -> Statevector:
        """Reconstruct a full state from the latent representation."""
        # Placeholder: simply pad with zeros
        padded = Statevector.from_label("0" * (self.circuit.num_qubits - self.num_latent))
        return latent.tensor(padded)

    def fidelity(self, a: Statevector, b: Statevector) -> float:
        return state_fidelity(a, b)


__all__ = [
    "GraphQNNAutoencoderQML",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "random_training_data",
    "state_fidelity",
]
