"""
Quantum graph neural network implemented with Pennylane.
Provides a parameter‑dependent unitary per edge and
uses state‑vector fidelity to build adjacency graphs.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import torch

Tensor = torch.Tensor

def _random_unitary(dim: int) -> Tensor:
    """Generate a random unitary matrix of given dimension."""
    mat = torch.randn(dim, dim, dtype=torch.cfloat)
    q, _ = torch.linalg.qr(mat)
    return q

def random_training_data(target_unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data by applying a target unitary to random states."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = target_unitary.shape[0]
    for _ in range(samples):
        state = torch.randn(dim, dtype=torch.cfloat)
        state = state / torch.linalg.norm(state)
        target = torch.matmul(target_unitary, state)
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random quantum GNN with the given architecture."""
    device = qml.device("default.qubit", wires=max(qnn_arch))
    params: List[torch.Tensor] = []
    layer_qnodes: List[qml.QNode] = []

    for num_inputs in qnn_arch[:-1]:
        # parameters for a single layer: one RY angle per qubit
        param = torch.randn(num_inputs, dtype=torch.float32, requires_grad=True)
        params.append(param)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def layer_qnode(p, state, n=num_inputs):
            qml.QubitStateVector(state, wires=range(n))
            for q in range(n):
                qml.RY(p[q], wires=q)
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
            return qml.state()

        layer_qnodes.append(layer_qnode)

    target_unitary = _random_unitary(2 ** qnn_arch[-1])

    # training data
    training_data = random_training_data(target_unitary, samples)

    return list(qnn_arch), params, training_data, target_unitary

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return squared fidelity between two pure states."""
    fid = torch.vdot(a, b)
    return float(fid.abs() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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

def feedforward(
    qnn_arch: Sequence[int],
    params: Sequence[torch.Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Forward propagation using a list of parameter tensors and a quantum device."""
    device = qml.device("default.qubit", wires=max(qnn_arch))
    layer_qnodes: List[qml.QNode] = []

    for num_inputs in qnn_arch[:-1]:
        @qml.qnode(device, interface="torch", diff_method="backprop")
        def layer_qnode(p, state, n=num_inputs):
            qml.QubitStateVector(state, wires=range(n))
            for q in range(n):
                qml.RY(p[q], wires=q)
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
            return qml.state()

        layer_qnodes.append(layer_qnode)

    outputs: List[List[Tensor]] = []

    for state, _ in samples:
        layer_states = [state]
        current_state = state
        for layer_qnode, param in zip(layer_qnodes, params):
            current_state = layer_qnode(param, current_state)
            layer_states.append(current_state)
        outputs.append(layer_states)

    return outputs

class GraphQNN:
    """
    Quantum graph neural network with a parameter‑dependent unitary per edge.
    """

    def __init__(self, qnn_arch: Sequence[int], device_name: str = "default.qubit"):
        self.arch = list(qnn_arch)
        self.device = qml.device(device_name, wires=max(qnn_arch))
        self.params: List[torch.Tensor] = []
        self.layer_qnodes: List[qml.QNode] = []

        for num_inputs in qnn_arch[:-1]:
            # one RY angle per qubit
            param = torch.randn(num_inputs, dtype=torch.float32, requires_grad=True)
            self.params.append(param)

            @qml.qnode(self.device, interface="torch", diff_method="backprop")
            def layer_qnode(p, state, n=num_inputs):
                qml.QubitStateVector(state, wires=range(n))
                for q in range(n):
                    qml.RY(p[q], wires=q)
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])
                return qml.state()

            self.layer_qnodes.append(layer_qnode)

    def forward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """
        Run the network on a batch of samples.
        :param samples: Iterable of (input_state, target_state) tuples.
        :return: List of state lists per sample, one state per layer (including input).
        """
        all_outputs: List[List[Tensor]] = []

        for state, _ in samples:
            layer_states = [state]
            current_state = state
            for layer_qnode, param in zip(self.layer_qnodes, self.params):
                current_state = layer_qnode(param, current_state)
                layer_states.append(current_state)
            all_outputs.append(layer_states)

        return all_outputs

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
