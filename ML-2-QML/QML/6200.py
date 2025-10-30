"""
GraphQNN – quantum side of a hybrid graph‑based neural network.

Features added:
* Variational circuit parameterised by qnn_arch.
* Random generation of a target unitary and training data.
* Feedforward that returns the state after each layer.
* Fidelity utilities identical to the classical side.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np  # deterministic NumPy backend

# --------------------------------------------------------------------
# 1.  Variational quantum circuit
# --------------------------------------------------------------------
def _param_shape(num_qubits: int) -> int:
    """Number of parameters for a layer of size num_qubits."""
    return num_qubits * 3  # Rx, Ry, Rz for each qubit

def _layer_circuit(num_qubits: int, params: np.ndarray) -> None:
    """Apply a layer of single‑qubit rotations followed by a CNOT chain."""
    idx = 0
    for q in range(num_qubits):
        qml.Rx(params[idx], wires=q)
        idx += 1
        qml.Ry(params[idx], wires=q)
        idx += 1
        qml.Rz(params[idx], wires=q)
        idx += 1
    for q in range(num_qubits - 1):
        qml.CNOT(wires=[q, q + 1])

def _variational_circuit(qnn_arch: Sequence[int], params: List[np.ndarray]) -> None:
    """Compose all layers of the variational circuit."""
    for num_qubits, layer_params in zip(qnn_arch, params):
        _layer_circuit(num_qubits, layer_params)

# --------------------------------------------------------------------
# 2.  Random data generation
# --------------------------------------------------------------------
def _random_state(num_qubits: int) -> np.ndarray:
    """Generate a random pure state of num_qubits qubits."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec

def random_training_data(
    unitary: np.ndarray,
    samples: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (state, target_state) pairs where target_state = U * state."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_state(num_qubits)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(
    qnn_arch: List[int],
    samples: int,
) -> Tuple[List[int], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Create a random target unitary and training data."""
    num_qubits = qnn_arch[-1]
    target_unitary = qml.utils.random_unitary(2 ** num_qubits)
    training_data = random_training_data(target_unitary, samples)

    # initialise parameters for each layer
    params: List[np.ndarray] = []
    for num in qnn_arch:
        params.append(np.random.randn(_param_shape(num)))

    return qnn_arch, params, training_data, target_unitary

# --------------------------------------------------------------------
# 3.  Feedforward
# --------------------------------------------------------------------
def feedforward(
    qnn_arch: Sequence[int],
    params: List[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Return the state after each layer for every sample."""
    dev = qml.device("default.qubit", wires=max(qnn_arch))

    @qml.qnode(dev, interface="autograd")
    def circuit(state: np.ndarray, layer_params: List[np.ndarray]) -> np.ndarray:
        qml.QubitStateVector(state, wires=range(len(state)))
        for num_qubits, layer_params in zip(qnn_arch, layer_params):
            _layer_circuit(num_qubits, layer_params)
        return qml.state()

    stored_states: List[List[np.ndarray]] = []
    for state, _ in samples:
        # compute full state
        final_state = circuit(state, params)
        # compute intermediate states by truncating the circuit
        intermediate: List[np.ndarray] = [state]
        for layer_idx, (num_qubits, layer_params) in enumerate(zip(qnn_arch, params)):
            @qml.qnode(dev, interface="autograd")
            def partial(state_in: np.ndarray, layer_params: np.ndarray) -> np.ndarray:
                qml.QubitStateVector(state_in, wires=range(num_qubits))
                _layer_circuit(num_qubits, layer_params)
                return qml.state()
            intermediate.append(partial(state, layer_params))
        stored_states.append(intermediate)
    return stored_states

# --------------------------------------------------------------------
# 4.  Fidelity utilities
# --------------------------------------------------------------------
def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared overlap between two pure states."""
    return float(np.abs(np.vdot(a, b)) ** 2)

def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
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
