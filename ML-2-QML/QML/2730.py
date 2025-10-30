"""UnifiedSamplerGraphQNN – quantum component.

This module implements a quantum sampler network that mirrors the
classical architecture.  It uses a parameterized quantum circuit
with a variational SamplerQNN from Qiskit Machine Learning and
provides graph‑based fidelity adjacency for the quantum states
produced during forward propagation.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
import torch
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

Tensor = qt.Qobj

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
    """Generate training data pairs (state, unitary*state)."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """Create a random layered unitary network and training data."""
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

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    """Propagate quantum states through the network."""
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

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class UnifiedSamplerGraphQNN:
    """Quantum sampler network with graph‑based fidelity adjacency."""

    def __init__(self, qnn_arch: List[int], graph_threshold: float = 0.9, secondary_threshold: float | None = None, backend: str | None = None):
        self.qnn_arch = qnn_arch
        self.graph_threshold = graph_threshold
        self.secondary_threshold = secondary_threshold

        # Build a parameterised circuit
        inputs2 = ParameterVector("input", len(qnn_arch[0]))
        weights2 = ParameterVector("weight", sum(qnn_arch[1:]))
        qc = QuantumCircuit(len(qnn_arch[0]))

        # Simple layered circuit mirroring the classical depth
        for idx, in_f in enumerate(qnn_arch[:-1]):
            out_f = qnn_arch[idx + 1]
            for q in range(in_f):
                qc.ry(inputs2[q], q)
            for q in range(out_f):
                qc.ry(weights2[idx * out_f + q], q)
            if out_f > 1:
                for q in range(out_f - 1):
                    qc.cx(q, q + 1)

        # SamplerQNN wrapper
        self.sampler = SamplerQNN(
            circuit=qc,
            input_params=inputs2,
            weight_params=weights2,
            sampler=StatevectorSampler(backend=backend)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the probability distribution given classical inputs."""
        param_values = inputs.detach().cpu().numpy()
        probs = self.sampler.run(param_values).probabilities
        return torch.tensor(probs, dtype=torch.float32)

    def build_graph(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> nx.Graph:
        """Construct a graph from the quantum state propagations."""
        states = [target for _, target in samples]
        graph = fidelity_adjacency(states, self.graph_threshold, secondary=self.secondary_threshold)
        for idx, state in enumerate(states):
            graph.nodes[idx]["state"] = state
        return graph

    def sample_from_graph(self, graph: nx.Graph, node: int) -> qt.Qobj:
        """Return the quantum state stored at the specified graph node."""
        return graph.nodes[node]["state"]

__all__ = [
    "UnifiedSamplerGraphQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
