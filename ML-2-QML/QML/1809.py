"""GraphQNN__gen409: Quantum graph neural network with spectral analysis.

This module mirrors the classical version but operates on quantum states
via Qutip.  It provides a `GraphQNN` class that can generate a random
quantum circuit, simulate forward propagation, and build a fidelity‑based
graph that optionally includes Laplacian eigen‑vectors for spectral
clustering.  The public API remains compatible with the original
functions.

Author: OpenAI GPT‑OSS-20b
"""

from __future__ import annotations

import itertools
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import qutip as qt
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F

QObj = qt.Qobj
Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1.  Utility functions  ----------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> QObj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> QObj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: QObj, source: int, target: int) -> QObj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> QObj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: QObj, samples: int) -> List[Tuple[QObj, QObj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[QObj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[QObj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: QObj, keep: Sequence[int]) -> QObj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: QObj, remove: Sequence[int]) -> QObj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[QObj]],
    layer: int,
    input_state: QObj,
) -> QObj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[QObj]],
    samples: Iterable[Tuple[QObj, QObj]],
) -> List[List[QObj]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: QObj, b: QObj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[QObj],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
    spectral: bool = False,
    k: int = 5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity above ``threshold`` receive weight 1.  An optional
    secondary threshold can add weaker edges.  When ``spectral`` is True,
    the first ``k`` Laplacian eigen‑vectors are attached to each node.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    if spectral:
        lap = nx.laplacian_matrix(graph).astype(float).todense()
        eigvals, eigvecs = np.linalg.eigh(lap)
        for idx in range(len(states)):
            graph.nodes[idx]["eigenvec"] = eigvecs[:, idx][:k].tolist()
    return graph


# --------------------------------------------------------------------------- #
# 2.  GraphQNN class -------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Quantum graph neural network utilities.

    The class mirrors the classical counterpart but operates on Qutip
    objects.  It exposes a ``random_network`` factory, a ``feedforward``
    method, and a ``HybridTrainer`` that can couple a classical GNN with a
    quantum circuit for end‑to‑end optimisation.
    """
    def __init__(self, qnn_arch: Sequence[int], device: str = "cpu"):
        self.qnn_arch = list(qnn_arch)
        self.device = device

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    def feedforward(self, unitaries: Sequence[Sequence[QObj]],
                    samples: Iterable[Tuple[QObj, QObj]]) -> List[List[QObj]]:
        return feedforward(self.qnn_arch, unitaries, samples)

    @staticmethod
    def state_fidelity(a: QObj, b: QObj) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[QObj],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
        spectral: bool = False,
        k: int = 5,
    ) -> nx.Graph:
        return fidelity_adjacency(
            states,
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
            spectral=spectral,
            k=k,
        )


# --------------------------------------------------------------------------- #
# 3.  Hybrid training helper ------------------------------------------------- #
# --------------------------------------------------------------------------- #
class HybridTrainer:
    """End‑to‑end optimisation of a classical GNN and a quantum layer."""

    def __init__(
        self,
        gnn: nn.Module,
        qlayer: nn.Module,
        lr: float = 1e-3,
    ):
        self.gnn = gnn
        self.qlayer = qlayer
        self.optimizer = torch.optim.Adam(
            list(gnn.parameters()) + list(qlayer.parameters()), lr=lr
        )

    def train_step(self, x: Tensor, y: Tensor) -> float:
        self.optimizer.zero_grad()
        z = self.gnn(x)
        q_out = self.qlayer(z)
        loss = F.mse_loss(q_out, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
