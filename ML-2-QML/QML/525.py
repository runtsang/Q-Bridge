"""Quantum graph‑based neural network with variational readout.

This module implements a variational quantum circuit that mirrors the
architecture defined by ``qnn_arch``.  Each layer is a collection of
parameterised single‑qubit rotations followed by a fixed
entanglement pattern.  The final measurement yields a probability
distribution over computational basis states that can be interpreted
as class probabilities.  The public API keeps the same helper
functions as the seed – ``random_network``, ``feedforward``,
``fidelity_adjacency`` – but now the underlying forward pass is
quantum.  A ``GraphQNN`` class provides convenient ``predict`` and
``train`` methods that wrap Pennylane's automatic differentiation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import pennylane as qml
import networkx as nx

Tensor = np.ndarray

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix of size 2^n."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q

def random_training_data(
    num_qubits: int,
    samples: int,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate supervised data by applying a random unitary to random states.

    The targets are one‑hot vectors corresponding to the basis state
    that the unitary would map the initial computational basis state
    to.
    """
    target_unitary = _random_qubit_unitary(num_qubits)
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        # start from a computational basis state
        idx = np.random.randint(0, 2 ** num_qubits)
        state = np.zeros(2 ** num_qubits, dtype=complex)
        state[idx] = 1.0
        # apply unitary
        transformed = target_unitary @ state
        # convert to one‑hot target
        target = np.zeros(2 ** num_qubits, dtype=float)
        target[np.argmax(np.abs(transformed))] = 1.0
        dataset.append((state, target))
    return dataset

def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> tuple[list[int], List[List[np.ndarray]], List[Tuple[Tensor, Tensor]], np.ndarray]:
    """Create a random variational architecture and training data.

    The returned ``params`` list contains a list of 3‑parameter
    rotation arrays for each gate in every layer.  The final parameter
    list corresponds to the readout layer and is ignored during
    training – the network learns to map input states to output
    probabilities directly.
    """
    arch = list(qnn_arch)
    num_qubits = arch[-1]
    params: List[List[np.ndarray]] = []
    # Hidden layers
    for layer_idx in range(1, len(arch)):
        layer_params: List[np.ndarray] = []
        for _ in range(arch[layer_idx]):
            layer_params.append(np.random.randn(3))  # Rot angles
        params.append(layer_params)
    # Readout parameters (not used in the circuit)
    readout_params = [np.random.randn(3) for _ in range(num_qubits)]
    params.append(readout_params)

    training_data = random_training_data(num_qubits, samples)
    target_unitary = _random_qubit_unitary(num_qubits)

    return arch, params, training_data, target_unitary

def _partial_trace(state: np.ndarray, keep: List[int]) -> np.ndarray:
    """Return the reduced density matrix over the qubits in ``keep``."""
    # For simplicity, we return the full state; a proper partial trace
    # would require reshaping and tracing out the other qubits.
    return state

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between two pure states."""
    return np.abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity greater than or equal to ``threshold`` receive
    weight 1.  When ``secondary`` is provided, fidelities between
    ``secondary`` and ``threshold`` are added with ``secondary_weight``.
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

class GraphQNN:
    """Variational quantum graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes.  The last element determines the number of qubits.
    params : List[List[np.ndarray]]
        Rotation angles for every gate in each layer.
    """

    def __init__(self, arch: Sequence[int], params: List[List[np.ndarray]]) -> None:
        self.arch = list(arch)
        self.params = params
        self.num_qubits = arch[-1]
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: List[List[np.ndarray]]) -> np.ndarray:
            # Encode input as angle embedding
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
            # Iterate layers
            for layer_idx, layer_params in enumerate(params):
                for qubit_idx, rot_params in enumerate(layer_params):
                    qml.Rot(*rot_params, wires=qubit_idx)
                # entangle with a simple linear chain
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # Return probability distribution
            return qml.probs(wires=range(self.num_qubits))
        return circuit

    def feedforward(
        self,
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> List[List[np.ndarray]]:
        """Run the variational circuit on the provided samples.

        Each element of the returned list corresponds to a sample and
        contains the probability vector after the final layer.
        """
        outputs: List[List[np.ndarray]] = []
        for state, _ in samples:
            probs = self._circuit(state, self.params)
            outputs.append([probs])
        return outputs

    def predict(
        self,
        inputs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return logits (unnormalised probabilities) and softmaxed probs."""
        probs = self._circuit(inputs, self.params)
        # Treat probs as logits for consistency with the classical API
        logits = probs
        softmax_probs = probs / probs.sum()
        return logits, softmax_probs

    def train(
        self,
        dataset: List[Tuple[np.ndarray, np.ndarray]],
        loss_fn=lambda logits, targets: -np.sum(targets * np.log(logits + 1e-12)),
        optimizer=None,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Simple training loop using Pennylane's autograd.

        Parameters
        ----------
        dataset : List[Tuple[np.ndarray, np.ndarray]]
            Each tuple is (input_state, target_one_hot).
        loss_fn : Callable
            Loss function that accepts logits and target distributions.
        optimizer : Callable
            Optimizer that accepts parameters and learning rate.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        """
        # Flatten all parameters into a single vector
        flat_params, unravel = qml.transforms.flatten_params(self.params)

        if optimizer is None:
            # Use a very simple gradient descent
            def optimizer(params, lr):
                return params - lr * params.grad
        for _ in range(epochs):
            def loss_fn_wrapped(flat_params):
                # Unravel to nested structure
                nested = unravel(flat_params)
                probs = self._circuit(dataset[0][0], nested)
                loss = loss_fn(probs, dataset[0][1])
                return loss
            grads = qml.grad(loss_fn_wrapped)(flat_params)
            flat_params = optimizer(flat_params, lr)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
