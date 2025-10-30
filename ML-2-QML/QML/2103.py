"""GraphQNN: Quantum module using Pennylane.

Key features:
* Variational circuit per layer with random unitary parameters.
* Automatic differentiation through Pennylane's QNode.
* Hybrid loss utilities to compare classical and quantum outputs.
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "hybrid_loss",
    "create_qnode",
    "qnode_to_state",
]


# --------------------------------------------------------------------- #
#  Utility helpers
# --------------------------------------------------------------------- #
def _random_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of size 2**num_qubits."""
    dim = 2**num_qubits
    mat = pnp.random.randn(dim, dim) + 1j * pnp.random.randn(dim, dim)
    q, _ = pnp.linalg.qr(mat)
    return q


def _random_state(num_qubits: int) -> np.ndarray:
    """Sample a random pure state vector."""
    dim = 2**num_qubits
    vec = pnp.random.randn(dim) + 1j * pnp.random.randn(dim)
    return vec / pnp.linalg.norm(vec)


# --------------------------------------------------------------------- #
#  Training data generation
# --------------------------------------------------------------------- #
def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (input_state, target_state) pairs."""
    data: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        inp = _random_state(unitary.shape[0].bit_length() - 1)
        tgt = unitary @ inp
        data.append((inp, tgt))
    return data


def random_network(arch: List[int], samples: int) -> Tuple[List[int], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Build a random variational network and create training data for its last layer."""
    num_qubits = arch[-1]
    target_unitary = _random_unitary(num_qubits)
    training = random_training_data(target_unitary, samples)

    # one unitary per layer (except input layer)
    unitaries: List[np.ndarray] = []
    for _ in range(1, len(arch)):
        unitaries.append(_random_unitary(arch[_]))
    return arch, unitaries, training, target_unitary


# --------------------------------------------------------------------- #
#  Forward propagation
# --------------------------------------------------------------------- #
def create_qnode(unitary: np.ndarray, dev_name: str = "default.qubit") -> qml.QNode:
    """Wrap a unitary matrix in a QNode that returns the state vector."""
    dev = qml.device(dev_name, wires=unitary.shape[0].bit_length() - 1)

    @qml.qnode(dev, interface="autograd")
    def circuit(state: np.ndarray):
        qml.QubitStateVector(state, wires=range(dev.num_wires))
        qml.UnitaryMatrix(unitary, wires=range(dev.num_wires))
        return qml.state()

    return circuit


def feedforward(
    arch: Sequence[int],
    unitaries: Sequence[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Apply each layer's unitary to the input state and return intermediate states."""
    outputs: List[List[np.ndarray]] = []
    for inp, _ in samples:
        states = [inp]
        current = inp
        for U in unitaries:
            current = U @ current
            states.append(current)
        outputs.append(states)
    return outputs


# --------------------------------------------------------------------- #
#  Fidelity utilities
# --------------------------------------------------------------------- #
def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between two pure states."""
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
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------- #
#  Hybrid utilities
# --------------------------------------------------------------------- #
def hybrid_loss(
    classical_preds: Sequence[np.ndarray],
    quantum_preds: Sequence[np.ndarray],
) -> float:
    """Mean squared error between classical and quantum predictions."""
    assert len(classical_preds) == len(quantum_preds)
    loss = 0.0
    for c, q in zip(classical_preds, quantum_preds):
        loss += np.mean((c - q) ** 2)
    return loss / len(classical_preds)


# --------------------------------------------------------------------- #
#  Helper to extract state from a QNode
# --------------------------------------------------------------------- #
def qnode_to_state(qnode: qml.QNode, state: np.ndarray) -> np.ndarray:
    """Return the state vector produced by a QNode."""
    return qnode(state)
