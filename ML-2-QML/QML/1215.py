"""Quantum GraphQNN implementation using Pennylane.

This module mirrors the classical API but replaces linear layers with
parameterised quantum circuits.  Each layer is a QNode that maps an
input state vector to an output state vector via a unitary
constructed from rotation gates and CNOT entanglement.  Training is
performed with Pennylane's autograd and an Adam optimiser.

Typical usage::

    arch, circuits, params, target = GraphQNN__gen156_qml.random_network_qml([2,3,2], 50)
    trained_params = GraphQNN__gen156_qml.train_variational_layer_qml(
        arch, circuits, params, 1, target, training_data, epochs=200
    )
    outputs = GraphQNN__gen156_qml.feedforward_qml(arch, circuits, params, training_data)
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Any

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# --------------------------------------------------------------------------- #
#   Utility functions
# --------------------------------------------------------------------------- #

def _random_state(num_qubits: int) -> np.ndarray:
    """Return a random pure state vector of size 2**num_qubits."""
    vec = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
    vec /= np.linalg.norm(vec)
    return vec

def _random_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q

def random_training_data(
    unitary: np.ndarray, samples: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate training pairs (state, unitary*state)."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = _random_state(unitary.shape[0].bit_length() - 1)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network_qml(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[qml.QNode], List[np.ndarray], np.ndarray]:
    """
    Create a random quantum network.

    Returns
    -------
    arch : list[int]
        Layer widths.
    circuits : list[QNode]
        One QNode per layer.
    params : list[np.ndarray]
        Trainable parameters for each layer.
    target_unitary : np.ndarray
        Unitary of the last layer (used for generating training data).
    """
    arch = list(qnn_arch)
    num_layers = len(arch) - 1
    circuits: List[qml.QNode] = []
    params: List[np.ndarray] = []

    # Device for simulation (CPU)
    dev = qml.device("default.qubit", wires=arch[-1])

    # Build layers
    for l in range(num_layers):
        in_f = arch[l]
        out_f = arch[l + 1]
        num_qubits = max(in_f, out_f)  # use the larger dimension

        # Parameter shape: (num_qubits, 3) for RX, RY, RZ
        layer_params = np.random.randn(num_qubits, 3)

        def circuit(state, params=layer_params, num_qubits=num_qubits):
            # Load state into qubits
            qml.QubitStateVector(state, wires=range(num_qubits))
            # Apply parameterised rotations
            for i in range(num_qubits):
                qml.RX(params[i, 0], wires=i)
                qml.RY(params[i, 1], wires=i)
                qml.RZ(params[i, 2], wires=i)
            # Entangle with CNOT chain
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure all qubits
            return qml.state()

        qnode = qml.QNode(circuit, dev, interface="autograd")
        circuits.append(qnode)
        params.append(layer_params)

    # Target unitary of the last layer
    target_unitary = _random_unitary(arch[-1])
    training_data = random_training_data(target_unitary, samples)

    return arch, circuits, params, target_unitary

def _apply_layer_qml(
    circuit: qml.QNode, params: np.ndarray, state: np.ndarray
) -> np.ndarray:
    """Run a single quantum layer on a state vector."""
    return circuit(state, params=params)

def feedforward_qml(
    qnn_arch: Sequence[int],
    circuits: Sequence[qml.QNode],
    params: Sequence[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Run the quantum network on the training samples."""
    stored: List[List[np.ndarray]] = []
    for state, _ in samples:
        activations = [state]
        current = state
        for circuit, param in zip(circuits, params):
            current = _apply_layer_qml(circuit, param, current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute squared overlap between two pure state vectors."""
    return float(np.abs(np.vdot(a, b)) ** 2)

def fidelity_adjacency(
    states: Sequence[np.ndarray],
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

# --------------------------------------------------------------------------- #
#   Variational training for a quantum layer
# --------------------------------------------------------------------------- #

def train_variational_layer_qml(
    qnn_arch: Sequence[int],
    circuits: List[qml.QNode],
    params: List[np.ndarray],
    layer_index: int,
    training_data: Iterable[Tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 200,
    lr: float = 0.01,
    device: str = "cpu",
) -> np.ndarray:
    """
    Train a single quantum layer to maximise fidelity with the target unitary.

    Parameters
    ----------
    qnn_arch : sequence of int
        Architecture of the network.
    circuits : list[QNode]
        Quantum circuits for each layer.
    params : list[np.ndarray]
        Current parameters for each layer.
    layer_index : int
        Index of the layer to train.
    training_data : iterable of (state, target_state) pairs
        Training samples where target_state is the desired output of the
        *next* layer (obtained from the target unitary).
    epochs : int
        Number of optimisation steps.
    lr : float
        Learning rate.
    device : str
        Device for Pennylane (default is CPU).
    """
    # Optimiser
    opt = qml.AdamOptimizer(stepsize=lr)

    # Parameters for the layer to train
    theta = params[layer_index]
    for _ in range(epochs):
        loss = 0.0
        for state, target in training_data:
            # Forward through previous layers
            h = state
            for i in range(layer_index):
                h = _apply_layer_qml(circuits[i], params[i], h)
            # Forward through the layer to train
            h_next = _apply_layer_qml(circuits[layer_index], theta, h)
            # Fidelity loss (negative)
            fid = state_fidelity(h_next, target)
            loss -= fid
        loss /= len(training_data)
        # Gradient step
        theta = opt.step(loss, theta)
    # Update the parameter array
    params[layer_index] = theta
    return theta

__all__ = [
    "feedforward_qml",
    "fidelity_adjacency",
    "random_network_qml",
    "random_training_data",
    "state_fidelity",
    "train_variational_layer_qml",
]
