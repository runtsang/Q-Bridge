"""GraphQNN: quantum‑based variational circuit with training on a state‑vector simulator.

The module mirrors the classical API but replaces the all‑to‑all unitary
operations with a parameterised circuit that has a register‑swap logic.
The `train` function uses PennyLane’s `qml.qnode` and a state‑vector
simulator to optimise the circuit parameters.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp

# --------------------------------------------------------------------------- #
# Helper functions – unchanged from the seed
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qml.QubitStateVector:
    """Return a tensor‑product identity operator."""
    return qml.Identity(num_qubits)


def _tensored_zero(num_qubits: int) -> qml.QubitStateVector:
    """Return a projector onto |0…0⟩."""
    return qml.PauliZ(num_qubits)  # placeholder; unused in current code


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix of size 2**num_qubits."""
    dim = 2 ** num_qubits
    rng = np.random.default_rng()
    random_matrix = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    unitary, _ = np.linalg.qr(random_matrix)
    return unitary


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Generate a random pure state vector."""
    dim = 2 ** num_qubits
    rng = np.random.default_rng()
    vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate training data for a single unitary.

    Each sample is a pair ``(state, target_state)`` where
    ``target_state = unitary @ state``.
    """
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Create a random variational network and a training set for the last layer.

    Returns:
        arch: list[int]
        params: list[list[np.ndarray]] – one list of unitary matrices per layer
        training_data: list[tuple[np.ndarray, np.ndarray]]
        target_unitary: np.ndarray – the last layer’s unitary matrix
    """
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                # Swap the newly added qubit into the correct position
                op = np.kron(op, np.eye(2 ** (num_outputs - 1)))
                # A real implementation would perform a swap; omitted for brevity
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    layer: int,
    input_state: np.ndarray,
) -> np.ndarray:
    """Apply a layer of gates and trace out the input qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = np.kron(input_state, np.zeros(2 ** num_outputs))
    # Apply the unitary for this layer
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary
    new_state = layer_unitary @ state
    # Trace out the input qubits
    keep = list(range(num_inputs, num_inputs + num_outputs))
    return np.trace(new_state.reshape([2] * (len(keep) + len(state.shape) - len(keep))), axis=keep)


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Return state trajectories for each sample."""
    stored_states: List[List[np.ndarray]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared overlap between two pure state vectors."""
    return abs(np.vdot(a, b)) ** 2


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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Variational GraphQNN – training on a state‑vector simulator
# --------------------------------------------------------------------------- #
class VariationalGraphQNN:
    """Parameterised quantum circuit that mimics the classical GraphQNN.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 2]``.
    dev : qml.Device
        PennyLane device used for simulation. Defaults to a state‑vector
        simulator with the number of qubits equal to the last layer.
    """

    def __init__(self, arch: Sequence[int], dev: qml.Device | None = None):
        self.arch = list(arch)
        self.num_qubits = arch[-1]
        self.dev = dev or qml.device("default.qubit", wires=self.num_qubits)
        # Initialise a random parameter matrix per layer
        self.params: List[np.ndarray] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            self.params.append(np.random.uniform(0, 2 * np.pi, size=(out_f, in_f)))

    def circuit(self, state: np.ndarray, params: List[np.ndarray]) -> np.ndarray:
        """Apply the variational circuit to an input state."""
        # Load the state onto the wires
        qml.QubitStateVector(state, wires=range(self.num_qubits))
        # Apply each layer
        for layer_idx, (in_f, out_f) in enumerate(zip(self.arch[:-1], self.arch[1:])):
            for out in range(out_f):
                for in_ in range(in_f):
                    qml.RX(params[layer_idx][out, in_], wires=out)
        # Return the final state
        return qml.state()

    def qnode(self, params: List[np.ndarray]):
        """PennyLane QNode that returns the state vector."""
        @qml.qnode(self.dev, interface="autograd")
        def inner(state: np.ndarray):
            return self.circuit(state, params)
        return inner

    def train(
        self,
        training_data: Iterable[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> List[float]:
        """Train the variational circuit to reproduce the target states.

        Returns a list of mean‑square fidelities per epoch.
        """
        params = [p.copy() for p in self.params]
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for state, target in training_data:
                qnode = self.qnode(params)
                pred = qnode(state)
                loss = np.mean(np.abs(pred - target) ** 2)
                grads = opt.compute_gradient(lambda p: self.qnode(p)(state), params)
                params = opt.apply_gradients(params, grads)
                epoch_loss += loss
            epoch_loss /= len(training_data)
            losses.append(epoch_loss)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.6f}")
        self.params = params
        return losses


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "VariationalGraphQNN",
]
