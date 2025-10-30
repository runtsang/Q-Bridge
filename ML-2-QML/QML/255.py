"""
GraphQNN: Quantum neural network utilities using Pennylane.

The quantum implementation mirrors the classical seed but replaces matrix
multiplications with variational circuits.  It provides:
* Random unitary generation and state preparation.
* A simple variational circuit that can be trained to approximate a target
  unitary via a fidelity loss.
* Feed‑forward execution of a stack of layers, each represented by a
  random unitary.
* A fidelity‑based graph construction routine identical to the classical
  version.
* A helper for computing a quantum kernel (state overlap).

All public names are kept compatible with the original seed.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy import linalg

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "quantum_kernel",
    "train_variational_circuit",
]


def _random_unitary(num_qubits: int) -> np.ndarray:
    """Generate a Haar‑random unitary on ``num_qubits``."""
    dim = 2**num_qubits
    matrix = pnp.random.normal(size=(dim, dim)) + 1j * pnp.random.normal(size=(dim, dim))
    unitary, _ = linalg.qr(matrix)
    return unitary


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Return a random pure state on ``num_qubits``."""
    dim = 2**num_qubits
    vec = pnp.random.normal(size=(dim, 1)) + 1j * pnp.random.normal(size=(dim, 1))
    vec /= linalg.norm(vec)
    return vec.squeeze()


def random_training_data(
    target_unitary: np.ndarray,
    samples: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate a dataset of random input states and their images under
    ``target_unitary``.
    """
    dataset: list[tuple[np.ndarray, np.ndarray]] = []
    num_qubits = int(np.log2(target_unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target = target_unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> tuple[list[int], list[list[np.ndarray]], list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Build a list of random unitary layers, a training set and a target unitary.

    Each layer is represented by a list containing a single unitary matrix
    that acts on the qubits of that layer.  For simplicity all layers use the
    same number of qubits as the final layer.
    """
    num_qubits = qnn_arch[-1]
    target_unitary = _random_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)

    layers: list[list[np.ndarray]] = [[]]  # dummy first entry to keep 1‑based indexing
    for _ in range(1, len(qnn_arch)):
        unitary = _random_unitary(num_qubits)
        layers.append([unitary])

    return list(qnn_arch), layers, training_data, target_unitary


def _layer_circuit(unitary: np.ndarray, num_qubits: int):
    """Return a Pennylane QNode that applies ``unitary`` to an arbitrary input
    state prepared by ``StatePreparation``.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(state: np.ndarray):
        qml.StatePreparation(state, wires=range(num_qubits))
        qml.QubitUnitary(unitary, wires=range(num_qubits))
        return qml.state()

    return circuit


def feedforward(
    qnn_arch: Sequence[int],
    layers: Sequence[Sequence[np.ndarray]],
    samples: Iterable[tuple[np.ndarray, np.ndarray]],
) -> list[list[np.ndarray]]:
    """Run a forward pass through the stack of unitary layers.

    For each sample the function returns a list containing the input state
    followed by the state after every layer.
    """
    num_qubits = qnn_arch[-1]
    stored_states: list[list[np.ndarray]] = []

    # Pre‑create the QNodes for each layer
    layer_circuits = [_layer_circuit(units[0], num_qubits) for units in layers[1:]]

    for state, _ in samples:
        layerwise = [state]
        current = state
        for circuit in layer_circuits:
            current = circuit(current)
            layerwise.append(current)
        stored_states.append(layerwise)

    return stored_states


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return float(np.abs(np.vdot(a, b)) ** 2)


def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive weight 1.
    When ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def quantum_kernel(
    state_a: np.ndarray,
    state_b: np.ndarray,
) -> float:
    """Return the kernel value between two pure states – the squared overlap."""
    return state_fidelity(state_a, state_b)


def train_variational_circuit(
    target_unitary: np.ndarray,
    num_qubits: int,
    num_layers: int,
    epochs: int = 200,
    lr: float = 0.05,
) -> np.ndarray:
    """Train a parameterised variational circuit to approximate ``target_unitary``.
    The circuit consists of alternating single‑qubit rotations and
    CNOT entangling gates.  The cost is 1 – fidelity between the
    circuit output and the target for a batch of random input states.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    # Parameterised circuit
    @qml.qnode(dev, interface="autograd")
    def variational_circuit(params: np.ndarray, state: np.ndarray):
        qml.StatePreparation(state, wires=range(num_qubits))
        for layer in range(num_layers):
            for q in range(num_qubits):
                # 3‑parameter RY,RZ,RX rotations
                qml.RY(params[layer, q, 0], wires=q)
                qml.RZ(params[layer, q, 1], wires=q)
                qml.RX(params[layer, q, 2], wires=q)
            # Entangle neighbouring qubits
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.state()

    # Initialise parameters
    rng = np.random.default_rng(seed=42)
    params = rng.normal(size=(num_layers, num_qubits, 3))

    opt = qml.GradientDescentOptimizer(stepsize=lr)

    # Training loop
    for _ in range(epochs):
        def cost_fn(p):
            loss = 0.0
            for _ in range(5):  # mini‑batch of 5 random states
                inp = _random_qubit_state(num_qubits)
                out = variational_circuit(p, inp)
                loss += 1.0 - quantum_kernel(out, target_unitary @ inp)
            return loss / 5.0

        params = opt.step(cost_fn, params)

    return params
