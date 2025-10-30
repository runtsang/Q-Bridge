"""GraphQNN: a quantum neural network based on Pennylane.

This module provides a variational circuit that mirrors the classical
feed‑forward network.  It supports data‑driven training, prediction, and
fidelity‑based graph construction.  The API matches the classical
implementation so that the two can be interchanged seamlessly.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import pennylane as qml

Tensor = np.ndarray

# --------------------------------------------------------------------------- #
# Core utilities
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Generate a random Haar‑distributed unitary."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(matrix)
    return q

def random_training_data(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Create a dataset of random pure states and their transformed versions."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        # random pure state
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        target = unitary @ vec
        dataset.append((vec, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random variational circuit and a synthetic training set."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # Initialise angles for each layer: 3 per qubit (Rot)
    params: List[Tensor] = []
    for l in range(1, len(qnn_arch)):
        num_wires = qnn_arch[l]
        num_params = num_wires * 3
        params.append(np.random.randn(num_params))
    return list(qnn_arch), params, training_data, target_unitary

# --------------------------------------------------------------------------- #
# Variational ansatz construction
# --------------------------------------------------------------------------- #
def _make_circuit(qnn_arch: Sequence[int], params: Sequence[Tensor]):
    """Return a Pennylane QNode that implements the variational ansatz."""
    wires = list(range(qnn_arch[-1]))

    @qml.qnode(qml.device("default.qubit", wires=wires), diff_method="backprop")
    def circuit(state: Tensor, *theta):
        # Prepare the input state
        qml.StatePrep(state, wires=wires)

        # Apply layers
        idx = 0
        for l in range(1, len(qnn_arch)):
            num_wires = qnn_arch[l]
            # Single‑qubit rotations
            for w in range(num_wires):
                a, b, c = theta[idx], theta[idx + 1], theta[idx + 2]
                qml.Rot(a, b, c, wires=w)
                idx += 3
            # Entangling layer
            for w in range(num_wires - 1):
                qml.CNOT(wires=(w, w + 1))
        return qml.State()
    return circuit

# --------------------------------------------------------------------------- #
# Forward propagation
# --------------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    params: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run the variational circuit on a batch of samples, returning the
    intermediate states (here only the final state is available)."""
    circuit = _make_circuit(qnn_arch, params)
    outputs: List[List[Tensor]] = []
    for state, _ in samples:
        final_state = circuit(state, *params)
        outputs.append([final_state])
    return outputs

# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two pure quantum states."""
    return float(np.abs(np.vdot(a, b)) ** 2)

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
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Training, prediction and evaluation
# --------------------------------------------------------------------------- #
def train(
    qnn_arch: Sequence[int],
    init_params: Sequence[Tensor],
    training_data: Iterable[Tuple[Tensor, Tensor]],
    target_unitary: Tensor,
    lr: float = 0.01,
    epochs: int = 200,
) -> List[Tensor]:
    """Optimise the variational parameters to reproduce `target_unitary`."""
    circuit = _make_circuit(qnn_arch, init_params)
    opt = qml.AdamOptimizer(stepsize=lr)

    # Flatten parameters into a single array for the optimiser
    flat_params = np.concatenate([p.flatten() for p in init_params])

    def loss_fn(flattened: Tensor) -> float:
        # Unpack into list of arrays matching init_params shapes
        ptr = 0
        theta_list: List[Tensor] = []
        for p in init_params:
            sz = p.size
            theta_list.append(flattened[ptr : ptr + sz].reshape(p.shape))
            ptr += sz

        loss = 0.0
        for state, _ in training_data:
            target_vec = target_unitary @ state
            pred_vec = circuit(state, *theta_list)
            loss += 1.0 - state_fidelity(pred_vec, target_vec)
        return loss / len(training_data)

    for _ in range(epochs):
        flat_params = opt.step(loss_fn, flat_params)

    # Re‑pack into list of arrays
    ptr = 0
    trained_params: List[Tensor] = []
    for p in init_params:
        sz = p.size
        trained_params.append(flat_params[ptr : ptr + sz].reshape(p.shape))
        ptr += sz
    return trained_params

def predict(
    qnn_arch: Sequence[int],
    params: Sequence[Tensor],
    state: Tensor,
) -> Tensor:
    """Evaluate the circuit on a single input state."""
    circuit = _make_circuit(qnn_arch, params)
    return circuit(state, *params)

def evaluate(
    qnn_arch: Sequence[int],
    params: Sequence[Tensor],
    training_data: Iterable[Tuple[Tensor, Tensor]],
    target_unitary: Tensor,
) -> float:
    """Return the mean fidelity loss on the training set."""
    circuit = _make_circuit(qnn_arch, params)
    loss = 0.0
    for state, _ in training_data:
        target_vec = target_unitary @ state
        pred_vec = circuit(state, *params)
        loss += 1.0 - state_fidelity(pred_vec, target_vec)
    return loss / len(training_data)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train",
    "predict",
    "evaluate",
]
