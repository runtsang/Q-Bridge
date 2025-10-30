"""Hybrid quantum graph‑sampler neural network.

The module defines a single class `HybridGraphSamplerQNN` that encapsulates
a parameterized sampler circuit, a chain of unitary layers, and graph
utilities based on state fidelities.  It mirrors the classical API
while leveraging Qiskit and qutip for quantum state manipulation.
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qiskit as qk
import qiskit.quantum_info as qi
import qiskit.quantum_info.states as qs
import qiskit_machine_learning.neural_networks as qnns
from qiskit.primitives import StatevectorSampler
from qiskit.circuit import QuantumCircuit, ParameterVector


def _random_qubit_unitary(num_qubits: int) -> qi.Operator:
    """Generate a Haar‑random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return qi.Operator(q)


def _random_qubit_state(num_qubits: int) -> qs.Statevector:
    """Create a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return qs.Statevector(vec)


def random_training_data(unitary: qi.Operator, samples: int) -> List[Tuple[qs.Statevector, qs.Statevector]]:
    """Generate pairs (|ψ⟩, U|ψ⟩) for training."""
    dataset: List[Tuple[qs.Statevector, qs.Statevector]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qi.Operator]], List[Tuple[qs.Statevector, qs.Statevector]], qi.Operator]:
    """Build a random layered unitary network and training set."""
    unitaries: List[List[qi.Operator]] = [[]]
    for layer in range(1, len(qnn_arch)):
        layer_ops: List[qi.Operator] = []
        for _ in range(qnn_arch[layer]):
            op = _random_qubit_unitary(qnn_arch[layer - 1] + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return qnn_arch, unitaries, training_data, target_unitary


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qi.Operator]], layer: int, input_state: qs.Statevector) -> qs.Statevector:
    """Apply the unitary gates of a single layer and trace out ancillas."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    ancilla_state = qs.Statevector(np.array([1, 0]) if num_outputs > 1 else [1])
    state = qi.Operator(qk.quantum_info.tensor([input_state, ancilla_state]))  # tensor product
    layer_unitary = qi.Operator(np.identity(state.data.shape[0], dtype=complex))
    for gate in unitaries[layer]:
        layer_unitary = gate @ layer_unitary
    new_state = layer_unitary @ state
    return qs.Statevector(new_state.data[:2 ** num_inputs])


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qi.Operator]],
    samples: Iterable[Tuple[qs.Statevector, qs.Statevector]],
) -> List[List[qs.Statevector]]:
    """Return the state after each layer for each sample."""
    all_states: List[List[qs.Statevector]] = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            states.append(current)
        all_states.append(states)
    return all_states


def state_fidelity(a: qs.Statevector, b: qs.Statevector) -> float:
    """Squared absolute overlap of two pure states."""
    return abs((a.data.conj().T @ b.data)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qs.Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def _build_sampler_circuit(qnn_arch: List[int]) -> QuantumCircuit:
    """Construct a parameterized circuit for the sampler branch."""
    qc = QuantumCircuit(qnn_arch[0])
    inp = ParameterVector("input", qnn_arch[0])
    wgt = ParameterVector("weight", sum(qnn_arch[1:]))
    for i, qubit in enumerate(range(qnn_arch[0])):
        qc.ry(inp[i], qubit)
    idx = 0
    for out in qnn_arch[1:]:
        for qubit in range(out):
            qc.ry(wgt[idx], qubit)
            idx += 1
    return qc


class HybridGraphSamplerQNN:
    """Hybrid quantum graph‑sampler network."""

    def __init__(self, qnn_arch: Sequence[int], samples: int = 100) -> None:
        self.arch, self.unitaries, self.training_data, self.target = random_network(list(qnn_arch), samples)
        self.sampler_circuit = _build_sampler_circuit(self.arch)
        self.sampler = StatevectorSampler()
        self.sampler_qnn = qnns.SamplerQNN(
            circuit=self.sampler_circuit,
            input_params=ParameterVector("input", self.arch[0]),
            weight_params=ParameterVector("weight", sum(self.arch[1:])),
            sampler=self.sampler,
        )

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """Return the probability distribution from the sampler circuit."""
        probs = self.sampler_qnn.predict(inputs.reshape(1, -1))
        return probs[0]

    def get_state_graph(self, threshold: float, *, secondary: float | None = None) -> nx.Graph:
        """Construct a fidelity graph from the last‑layer states of the training set."""
        final_states = [states[-1] for states in feedforward(self.arch, self.unitaries, self.training_data)]
        return fidelity_adjacency(final_states, threshold, secondary=secondary)


__all__ = [
    "HybridGraphSamplerQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
