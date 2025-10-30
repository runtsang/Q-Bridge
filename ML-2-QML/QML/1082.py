"""Quantum graph neural network utilities using PennyLane.

This module mirrors the classical seed but replaces linear
transformations with parameterised quantum circuits.  The
implementation introduces:

* :class:`GraphQNNQuantum` – a variational circuit builder that
  accepts an architecture list.  Each layer consists of
  single‑qubit rotations followed by CNOT entangling gates.
* :func:`train_qnn` – a simple training loop that optimises the
  parameters to minimise the mean‑squared error between the
  circuit output state and a target unitary.
* :func:`random_network` – generates a target unitary and a
  matching training dataset.
* :func:`random_training_data` – produces input states and the
  corresponding target states.
* :func:`state_fidelity` – computes the absolute squared overlap
  between two pure states.
* :func:`fidelity_adjacency` – builds a weighted graph from state
  fidelities.

The public API is intentionally compatible with the classical
module to allow side‑by‑side experimentation.

"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union

import networkx as nx
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _random_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary by QR decomposition of a random matrix."""
    dim = 2 ** num_qubits
    mat = pnp.random.randn(dim, dim) + 1j * pnp.random.randn(dim, dim)
    q, _ = pnp.linalg.qr(mat)
    return q


def random_training_data(
    target_unitary: np.ndarray,
    samples: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate input states and their images under the target unitary."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    num_qubits = int(np.log2(target_unitary.shape[0]))
    for _ in range(samples):
        vec = pnp.random.randn(2 ** num_qubits) + 1j * pnp.random.randn(2 ** num_qubits)
        vec /= pnp.linalg.norm(vec)
        input_state = vec
        output_state = target_unitary @ vec
        dataset.append((input_state, output_state))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a target unitary for the last layer and a training dataset."""
    num_qubits = qnn_arch[-1]
    target_unitary = _random_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), training_data, target_unitary


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[Union[np.ndarray, "torch.Tensor"]],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.
    Supports both quantum states and classical tensors."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        if isinstance(state_i, np.ndarray):
            fid = state_fidelity(state_i, state_j)
        else:
            a = state_i / (np.linalg.norm(state_i) + 1e-12)
            b = state_j / (np.linalg.norm(state_j) + 1e-12)
            fid = abs(np.dot(a, b)) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Variational circuit class
# --------------------------------------------------------------------------- #
class GraphQNNQuantum:
    """Variational quantum circuit that implements a graph‑based
    neural network.  Each layer consists of parameterised rotations
    followed by a full‑CNOT entangling pattern."""

    def __init__(self, architecture: Sequence[int], dev: qml.Device | None = None):
        self.architecture = list(architecture)
        self.num_qubits = self.architecture[-1]
        self.dev = dev or qml.device("default.qubit", wires=self.num_qubits)
        self.params = self._initialize_params()

    def _initialize_params(self) -> np.ndarray:
        """Create a parameter array for all layers."""
        param_list = []
        for out_f in self.architecture[1:]:
            # 3 rotation angles per qubit per layer
            param_list.append(np.random.randn(out_f, 3))
        return np.concatenate([p.ravel() for p in param_list])

    def _circuit(self, params: np.ndarray, input_state: np.ndarray):
        """PennyLane QNode that applies the variational circuit to an input state."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            qml.StatePrep(input_state, wires=range(self.num_qubits))
            idx = 0
            for out_f in self.architecture[1:]:
                layer_params = params[idx : idx + out_f * 3].reshape((out_f, 3))
                idx += out_f * 3
                # single‑qubit rotations
                for qubit, (rx, ry, rz) in enumerate(layer_params):
                    qml.RX(rx, wires=qubit)
                    qml.RY(ry, wires=qubit)
                    qml.RZ(rz, wires=qubit)
                # full‑CNOT entanglement
                for i in range(out_f - 1):
                    qml.CNOT(wires=(i, i + 1))
            return qml.state()
        return circuit()

    def feedforward(
        self,
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> List[np.ndarray]:
        """Run the circuit on each input state and return the output state."""
        outputs = []
        for input_state, _ in samples:
            output_state = self._circuit(self.params, input_state)
            outputs.append(output_state)
        return outputs

    def loss(self, params: np.ndarray, batch: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Mean‑squared error between circuit output and target states."""
        loss = 0.0
        for input_state, target_state in batch:
            output_state = self._circuit(params, input_state)
            loss += np.sum(np.abs(output_state - target_state) ** 2)
        return loss / len(batch)

    def train(
        self,
        training_data: Iterable[Tuple[np.ndarray, np.ndarray]],
        *,
        epochs: int = 10,
        lr: float = 0.01,
    ) -> List[float]:
        """Simple training loop using gradient descent."""
        params = self.params.copy()
        losses: List[float] = []
        for epoch in range(epochs):
            loss_val = self.loss(params, training_data)
            grads = qml.grad(self.loss)(params, training_data)
            params -= lr * grads
            losses.append(loss_val)
        self.params = params
        return losses


def train_qnn(
    qnn: GraphQNNQuantum,
    training_data: Iterable[Tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 10,
    lr: float = 0.01,
) -> List[float]:
    """Convenience wrapper around :meth:`GraphQNNQuantum.train`."""
    return qnn.train(training_data, epochs=epochs, lr=lr)


__all__ = [
    "GraphQNNQuantum",
    "train_qnn",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
