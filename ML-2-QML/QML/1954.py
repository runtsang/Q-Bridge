"""GraphQNN__gen149: Quantum‑classical hybrid network with variational layers and graph diagnostics.

This module extends the original QNN utilities by providing a variational circuit that can be trained to emulate a classical feed‑forward network.
It exposes:
- a `QNode` that implements a configurable depth of variational layers.
- a helper to generate random training data (input states and target states).
- a fidelity‑based loss and a training routine that uses PennyLane's gradient‑based optimisers.
- a decorator that records the fidelity‑based adjacency graph of the intermediate states for visualisation.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx
import pennylane as qml
import numpy as np

# --------------------------------------------------------------------------- #
#  Core quantum utilities – unchanged from the seed
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qml.Device:
    """Return a PennyLane device that provides an identity operation on ``num_qubits``."""
    return qml.device("default.qubit", wires=num_qubits)

def _random_qubit_state(num_qubits: int) -> qml.Statevector:
    """Generate a random pure state on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return qml.Statevector(vec, wires=range(num_qubits))

def _random_qubit_unitary(num_qubits: int) -> qml.QNode:
    """Return a QNode that implements a random unitary on ``num_qubits`` qubits."""
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(*params):
        for i in range(num_qubits):
            qml.Rot(*params[i*3:(i+1)*3], wires=i)
        return qml.state()
    init_params = np.random.normal(size=(num_qubits * 3,))
    return circuit, init_params

def random_training_data(num_qubits: int, samples: int) -> list[tuple[qml.Statevector, qml.Statevector]]:
    """Generate training data by applying a random unitary to random states."""
    circuit, init_params = _random_qubit_unitary(num_qubits)
    dataset: list[tuple[qml.Statevector, qml.Statevector]] = []
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target = circuit(*init_params)
        dataset.append((state, target))
    return dataset

# --------------------------------------------------------------------------- #
#  Variational network construction
# --------------------------------------------------------------------------- #

def random_network(qnn_arch: list[int], samples: int):
    """Construct a variational circuit that mimics a network defined by ``qnn_arch``."""
    depth = len(qnn_arch) - 1
    num_qubits = qnn_arch[-1]
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(*params):
        idx = 0
        for layer in range(depth):
            for n in range(qnn_arch[layer + 1]):
                qml.Rot(
                    params[idx], params[idx + 1], params[idx + 2],
                    wires=layer
                )
                idx += 3
            if layer < depth - 1:
                qml.CNOT(wires=[layer, layer + 1])
        return qml.state()

    init_params = np.random.normal(size=circuit.num_params)
    target_circuit, _ = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(num_qubits, samples)
    return qnn_arch, circuit, init_params, training_data, target_circuit

# --------------------------------------------------------------------------- #
#  Forward propagation
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    circuit: qml.QNode,
    init_params: np.ndarray,
    samples: Iterable[tuple[qml.Statevector, qml.Statevector]],
) -> list[list[qml.Statevector]]:
    """Run the variational circuit on each input state and record intermediate states."""
    stored_states: list[list[qml.Statevector]] = []
    for state, _ in samples:
        final_state = circuit(*init_params)
        stored_states.append([state, final_state])
    return stored_states

# --------------------------------------------------------------------------- #
#  Graph utility functions
# --------------------------------------------------------------------------- #

def state_fidelity(a: qml.Statevector, b: qml.Statevector) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs(a.data.conj().T @ b.data) ** 2

def fidelity_adjacency(
    states: Sequence[qml.Statevector],
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

# --------------------------------------------------------------------------- #
#  Training utilities – new additions
# --------------------------------------------------------------------------- #

def fidelity_loss(output: qml.Statevector, target: qml.Statevector) -> float:
    """Return 1 - fidelity between two statevectors."""
    return 1.0 - state_fidelity(output, target)

def train_variational(
    circuit: qml.QNode,
    init_params: np.ndarray,
    data: Iterable[tuple[qml.Statevector, qml.Statevector]],
    optimizer: qml.Optimizer,
    epochs: int = 200,
    log_interval: int = 20,
) -> list[float]:
    """Train a variational circuit to match target states using fidelity loss."""
    params = init_params
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for state, target in data:
            def cost_func(p):
                out_state = circuit(*p)
                return fidelity_loss(out_state, target)
            grads = qml.grad(cost_func)(params)
            params = optimizer.step(cost_func, params)
            epoch_loss += cost_func(params)
        if epoch % log_interval == 0:
            losses.append(epoch_loss / len(data))
    return losses

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_variational",
    "fidelity_loss",
]
