"""Hybrid Graph Quantum Neural Network – Quantum side.

The quantum module mirrors the original seed but now includes a
variational‑parameterised circuit that is trained by the classical
optimiser in `HybridGraphQNN`.  The implementation uses PennyLane
to define the circuit, automatically differentiable with respect to
classical parameters, and exposes the same API as the original module.

"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import pennylane as qml
import numpy as np
import torch

Tensor = torch.Tensor


def _random_qubit_unitary(num_qubits: int) -> qml.QNode:
    """Return a random unitary as a PennyLane QNode."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(x: Tensor, params: Tensor):
        # Random single‑qubit rotations and entangling CNOTs
        for i in range(num_qubits):
            qml.RX(params[i], wires=i)
            qml.RY(params[i + num_qubits], wires=i)
            qml.RZ(params[i + 2 * num_qubits], wires=i)
        for i in range(num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    return circuit


def random_training_data(unitary: qml.QNode, samples: int) -> List[tuple[Tensor, Tensor]]:
    """Generate (state, unitary(state)) pairs for training."""
    dataset: List[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = torch.randn(2 ** unitary.wires, dtype=torch.complex64)
        state = state / torch.norm(state)
        target = unitary(state, torch.randn(3 * unitary.wires))  # dummy params
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Return architecture, list of QNodes, training data and the target unitary."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    qnodes: List[List[qml.QNode]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qml.QNode] = []
        for out in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                # Pad with identity on the extra outputs
                op = qml.operation.expand(op, wires=list(range(num_inputs + 1)) + list(range(num_inputs + 1, num_outputs)))
            layer_ops.append(op)
        qnodes.append(layer_ops)

    return qnn_arch, qnodes, training_data, target_unitary


def _layer_channel(
    qnn_arch: Sequence[int],
    qnodes: Sequence[Sequence[qml.QNode]],
    layer: int,
    input_state: Tensor,
) -> Tensor:
    """Apply a layer of QNodes to an input state and return the reduced output."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = torch.cat([input_state, torch.zeros(2 ** (num_outputs - 1))], dim=0)
    for gate in qnodes[layer]:
        state = gate(state, torch.randn(3 * gate.wires))
    # Partial trace: keep only the first num_outputs qubits
    keep = list(range(num_outputs))
    return state[keep]


def feedforward(
    qnn_arch: Sequence[int],
    qnodes: Sequence[Sequence[qml.QNode]],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return a list of state trajectories for each sample."""
    trajectories: List[List[Tensor]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, qnodes, layer, current)
            layerwise.append(current)
        trajectories.append(layerwise)
    return trajectories


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two pure states."""
    return float((torch.conj(a) * b).abs().sum() ** 2)


def fidelity_adjacency(
    states: List[Tensor],
    threshold: float,
    *,
    secondary: None | float = None,
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


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
