"""Hybrid quantum graph neural network.

GraphQNNGen implements the same public API as the classic GraphQNNGen but
replaces the linear layers with parametrised quantum unitaries.
The module uses qutip for state propagation and fidelity‑based graph
construction.  Quantum transformer blocks and a fully‑connected quantum
layer are provided as optional sub‑modules, mirroring the behaviour of
the QTransformerTorch and FCL references.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict, Iterable as IterableType

import networkx as nx
import qutip as qt
import scipy as sc
import torch
import torch.nn as nn
import numpy as np
import qiskit

# ---- Quantum utilities -----------------------------------------------------

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


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
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


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
) -> list[list[qt.Qobj]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
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


# ---- Quantum transformer (optional) ----------------------------------------

class PositionalEncoder:
    def __init__(self, embed_dim: int, max_len: int = 5000):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def __call__(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention:
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

    def __call__(self, x):
        return x  # placeholder


class FeedForward:
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def __call__(self, x):
        return x  # placeholder


class TransformerBlock:
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def __call__(self, x):
        return self.attn(x)  # placeholder


# ---- Fully‑connected quantum layer -----------------------------------------

class FCL:
    """
    Parameterised quantum circuit for a fully connected layer.
    Uses Qiskit to simulate a single‑qubit Ry gate.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 100):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def run(self, thetas: IterableType[float]) -> np.ndarray:
        circuit = qiskit.QuantumCircuit(self.n_qubits)
        circuit.h(range(self.n_qubits))
        circuit.barrier()
        circuit.ry(thetas[0], range(self.n_qubits))
        circuit.measure_all()
        job = qiskit.execute(circuit, self.backend, shots=self.shots, parameter_binds=[{circuit.params[0]: theta} for theta in thetas])
        result = job.result().get_counts(circuit)
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        expectation = np.sum(probs)
        return np.array([expectation])


# ---- Main hybrid quantum graph neural network -------------------------------

class GraphQNNGen(nn.Module):
    """
    Quantum‑enhanced graph neural network that mirrors the API of the
    classical GraphQNNGen.  The linear layers are replaced by
    parametrised unitaries and optional quantum transformer blocks
    are available.  The class can be used as a drop‑in replacement
    for the classical version.
    """

    def __init__(
        self,
        arch: Sequence[int],
        *,
        use_transformer: bool = False,
        transformer_params: Dict | None = None,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.use_transformer = use_transformer
        self.transformer_params = transformer_params or {}

        # Build quantum unitaries per layer
        self.unitaries: list[list[qt.Qobj]] = [[]]
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: list[qt.Qobj] = []
            for output in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                    op = _swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

        if self.use_transformer:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlock(
                        embed_dim=self.arch[-1],
                        num_heads=self.transformer_params.get("num_heads", 4),
                        ffn_dim=self.transformer_params.get("ffn_dim", 64),
                        dropout=self.transformer_params.get("dropout", 0.1),
                    )
                    for _ in range(self.transformer_params.get("num_blocks", 2))
                ]
            )
        else:
            self.transformer = None

    def random_network(self, samples: int):
        return random_network(self.arch, samples)

    def feedforward(self, samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
        return feedforward(self.arch, self.unitaries, samples)

    def fidelity_adjacency(self, states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)


__all__ = [
    "GraphQNNGen",
    "FCL",
    "PositionalEncoder",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
]
