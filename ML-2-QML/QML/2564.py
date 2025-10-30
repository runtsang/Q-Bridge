"""Quantum graph autoencoder that embeds classical graph adjacency into a variational circuit and
reconstructs it via swap‑test fidelity.  The module mirrors the classical
GraphQNN_AE but replaces the linear layers with a quantum ansatz."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Sequence as SequenceType

import networkx as nx
import numpy as np
import qiskit as qk
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN

Tensor = qt.Qobj


# --------------------------------------------------------------------------- #
# 1. Quantum utilities (adapted from seed)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 2. Quantum autoencoder circuit
# --------------------------------------------------------------------------- #
def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()
    auxiliary = num_latent + 2 * num_trash
    qc.h(auxiliary)
    for i in range(num_trash):
        qc.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
    qc.h(auxiliary)
    qc.measure(auxiliary, cr[0])
    return qc


def AutoencoderQNN(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Return a SamplerQNN configured with the autoencoder circuit."""
    sampler = StatevectorSampler()
    qc = _auto_encoder_circuit(num_latent, num_trash)
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


# --------------------------------------------------------------------------- #
# 3. Hybrid quantum GraphQNN_AE
# --------------------------------------------------------------------------- #
class GraphQNN_AE:
    """
    Quantum graph autoencoder that embeds classical graph adjacency into a variational
    circuit and reconstructs the graph via a fidelity‑based adjacency graph.
    The class mimics the classical GraphQNN_AE interface but uses a quantum ansatz.
    """

    def __init__(self, qnn_arch: Sequence[int], num_trash: int = 2) -> None:
        self.qnn_arch = list(qnn_arch)
        self.num_trash = num_trash
        self.circuit = _auto_encoder_circuit(qnn_arch[0], num_trash)
        self.qnn = AutoencoderQNN(num_latent=qnn_arch[0], num_trash=num_trash)

    def encode(self, graph: nx.Graph) -> List[qt.Qobj]:
        """Encode each node of the input graph into a quantum state."""
        states: List[qt.Qobj] = []
        for node in graph.nodes:
            deg = graph.degree[node]
            state = _random_qubit_state(1)
            # Simple encoding of degree (placeholder)
            state.data[0] = deg / (graph.number_of_nodes() + 1)
            states.append(state)
        return states

    def forward(self, graph: nx.Graph) -> nx.Graph:
        """
        Forward pass: encode → variational circuit → fidelity‑based adjacency reconstruction.
        """
        latent_states = self.encode(graph)
        # Simulate the circuit for each latent state
        sampled_states = [self.qnn.forward(state) for state in latent_states]
        # Build adjacency from fidelities
        return fidelity_adjacency(sampled_states, threshold=0.8, secondary=0.6)


__all__ = [
    "GraphQNN_AE",
    "AutoencoderQNN",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
