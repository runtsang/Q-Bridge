"""Quantum‑graph neural network using PennyLane.

This module extends the original seed by providing a single
`GraphQNN__gen287` class.  The class can be instantiated with a graph‑like
architecture (list[int]) and a training mode (``'mse'`` or ``'hybrid'``).
When ``'hybrid'`` the loss is a weighted sum of
classical mean‑square error **and** the quantum fidelity between the
final state and a target state.  The class exposes three public
methods:

* ``train`` – performs stochastic gradient descent on the
  variational parameters using PennyLane’s autograd.
* ``graph`` – returns a NetworkX graph built from the layer‑wise
  output states.
* ``predict`` – forwards a single batch of inputs and returns the
  raw output states.

The implementation keeps the original helper functions
(`feedforward`, ``state_fidelity`` and ``fidelity_adjacency``)
so that downstream tests can still import them directly.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np

# --------------------------------------------------------------------------- #
# Helper utilities – quantum variants
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix of size 2**num_qubits."""
    dim = 2**num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    # QR decomposition to get a unitary
    q, _ = np.linalg.qr(mat)
    return q


def random_training_data(
    target_state: np.ndarray,
    samples: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate random input states and their transformed targets."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = target_state.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        dataset.append((state, target_state @ state))
    return dataset


def random_network(arch: List[int], samples: int):
    """Create a random variational network and synthetic training data.

    The architecture is expected to contain a *single* qubit count that is
    repeated across all layers.  This restriction keeps the implementation
    simple and guarantees that the input and output state dimensions
    match across layers.
    """
    num_qubits = arch[0]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    # Store parameters: one array per layer
    params: List[np.ndarray] = []
    for _ in arch:
        # StronglyEntanglingLayers requires 2 * layers * 3 parameters
        layers = 1
        param_shape = (layers, 2, 3)
        params.append(np.random.randn(*param_shape))
    return arch, params, training_data, target_unitary


def feedforward(
    arch: Sequence[int],
    params: Sequence[np.ndarray],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Run the quantum network on the provided samples."""
    stored: List[List[np.ndarray]] = []
    for state, _ in samples:
        activations = [state]
        current = state
        for num_qubits, theta in zip(arch, params):
            dev = qml.device("default.qubit", wires=num_qubits, shots=0)

            @qml.qnode(dev, interface="autograd")
            def circuit(x, t):
                qml.StatePrep(x, wires=range(num_qubits))
                qml.StronglyEntanglingLayers(t, wires=range(num_qubits))
                return qml.state()

            current = circuit(current, theta)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute squared overlap between two pure quantum states."""
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
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# GraphQNN__gen287 class – quantum variational network
# --------------------------------------------------------------------------- #
class GraphQNN__gen287:
    """Quantum graph neural network using PennyLane.

    Parameters
    ----------
    arch : Sequence[int]
        Number of qubits per layer.  All elements must be equal to keep the
        input and output state dimensions identical across layers.
    loss_type : str, optional
        Either ``'mse'`` (default) or ``'hybrid'``.
    device : str, optional
        PennyLane device name, e.g. ``'default.qubit'``.
    """

    def __init__(
        self,
        arch: Sequence[int],
        loss_type: str = "mse",
        device: str = "default.qubit",
    ) -> None:
        self.arch = list(arch)
        self.loss_type = loss_type.lower()
        self.device_name = device
        # Create trainable parameters per layer
        self.params: List[np.ndarray] = []
        self.qnodes: List[Callable] = []
        for num_qubits in self.arch:
            layers = 1
            theta = np.random.randn(layers, 2, 3)
            self.params.append(theta)
            dev = qml.device(self.device_name, wires=num_qubits, shots=0)

            @qml.qnode(dev, interface="autograd")
            def circuit(x, t, num_qubits=num_qubits):
                qml.StatePrep(x, wires=range(num_qubits))
                qml.StronglyEntanglingLayers(t, wires=range(num_qubits))
                return qml.state()

            self.qnodes.append(circuit)

    def predict(self, input_state: np.ndarray) -> List[np.ndarray]:
        """Forward pass returning layer‑wise output states."""
        activations = [input_state]
        current = input_state
        for qnode, theta in zip(self.qnodes, self.params):
            current = qnode(current, theta)
            activations.append(current)
        return activations

    def predict_with_params(
        self,
        input_state: np.ndarray,
        params: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """Forward pass using an explicit list of parameters."""
        activations = [input_state]
        current = input_state
        for qnode, theta in zip(self.qnodes, params):
            current = qnode(current, theta)
            activations.append(current)
        return activations

    def _loss(self, predictions: List[np.ndarray], target: np.ndarray) -> float:
        """Compute loss for a single sample."""
        pred = predictions[-1]
        mse = np.mean((pred - target) ** 2)
        if self.loss_type == "hybrid":
            fid = state_fidelity(pred, target)
            fid_loss = 1.0 - fid
            return mse + 0.5 * fid_loss
        return mse

    def train(
        self,
        data: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> List[float]:
        """Train the variational parameters using gradient descent.

        Returns a list of the loss value after each epoch.
        """
        history: List[float] = []

        def loss_fn(*params):
            loss = 0.0
            for x, y in data:
                act = self.predict_with_params(x, params)
                loss += self._loss(act, y)
            return loss / len(data)

        grad_fn = qml.grad(loss_fn)

        for epoch in range(epochs):
            grads = grad_fn(*self.params)
            # Update parameters
            self.params = [p - lr * g for p, g in zip(self.params, grads)]
            # Record loss for this epoch
            epoch_loss = loss_fn(*self.params)
            history.append(epoch_loss)
        return history

    def graph(self, activations: List[List[np.ndarray]], threshold: float) -> nx.Graph:
        """Build a fidelity‑based adjacency graph from final states."""
        final_states = [act[-1] for act in activations]
        return fidelity_adjacency(final_states, threshold)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "GraphQNN__gen287",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
]
