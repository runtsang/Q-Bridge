"""Quantum Graph Neural Network with variational layers and fidelity‑based training.

Key extensions over the original QML seed:
* Parameterized variational circuit per layer using PennyLane.
* Monte‑Carlo fidelity loss with early‑stopping.
* Unified ``GraphQNNGen444QML`` class mirroring the classical API.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import pennylane as qml
import networkx as nx

# -------------------------------------------------------------------------- #
# Utility functions
# -------------------------------------------------------------------------- #
def _random_qubit_state(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate (input_state, target_state) pairs for a given target unitary."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        inp = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        tgt = unitary @ inp
        dataset.append((inp, tgt))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random parameterised circuit, training data and the target unitary."""
    max_wires = max(qnn_arch)
    dev = qml.device("default.qubit", wires=max_wires)

    # Target unitary on the largest layer
    target_unitary = qml.math.random_unitary(2 ** max_wires)
    training_data = random_training_data(target_unitary, samples)

    # Parameterised gates per layer
    params: List[np.ndarray] = []
    for layer in range(1, len(qnn_arch)):
        num_out = qnn_arch[layer]
        # 3 parameters per output wire (Rot gate)
        params.append(np.random.uniform(0, 2 * np.pi, size=(num_out, 3)))

    return list(qnn_arch), params, training_data, target_unitary


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between two pure state vectors."""
    return np.abs(np.vdot(a, b)) ** 2


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


# -------------------------------------------------------------------------- #
# GraphQNNGen444QML
# -------------------------------------------------------------------------- #
class GraphQNNGen444QML:
    """Hybrid quantum Graph Neural Network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer widths, including input and output sizes.
    params : List[np.ndarray] | None, optional
        Pre‑initialized variational parameters. If ``None`` random parameters are created.
    """

    def __init__(
        self,
        arch: Sequence[int],
        params: List[np.ndarray] | None = None,
    ):
        self.arch = list(arch)
        self.max_wires = max(arch)
        self.dev = qml.device("default.qubit", wires=self.max_wires)
        self.params = params if params is not None else [
            np.random.uniform(0, 2 * np.pi, size=(arch[layer + 1], 3))
            for layer in range(len(arch) - 1)
        ]
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: List[np.ndarray]):
            """Parameterized variational circuit."""
            qml.StatePrep(inputs, wires=range(self.max_wires))
            for layer_idx, layer_params in enumerate(params):
                num_out = self.arch[layer_idx + 1]
                for wire in range(num_out):
                    a, b, c = layer_params[wire]
                    qml.Rot(a, b, c, wires=wire)
            return qml.state()

        return circuit

    def forward(self, inputs: np.ndarray) -> List[np.ndarray]:
        """Return list of state vectors per layer (only final state is returned)."""
        state = inputs
        states: List[np.ndarray] = [state]
        for layer_params in self.params:
            state = self._circuit(state, [layer_params])
            states.append(state)
        return states

    def loss(self, outputs: List[np.ndarray], threshold: float = 0.8) -> float:
        """Graph‑based smoothness loss on the final state."""
        final = outputs[-1]
        graph = fidelity_adjacency([final], threshold)
        loss_val = 0.0
        for u, v, data in graph.edges(data=True):
            fid = state_fidelity(final, final)  # identical, placeholder
            loss_val += (1.0 - fid) * data["weight"]
        return loss_val

    def train(
        self,
        samples: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 200,
        lr: float = 0.01,
        val_samples: List[Tuple[np.ndarray, np.ndarray]] | None = None,
        patience: int = 10,
        threshold: float = 0.8,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Train the variational circuit using gradient descent with early stopping.

        Returns
        -------
        params : List[np.ndarray]
            Final trained parameters.
        history : List[float]
            Training loss history.
        """
        history: List[float] = []
        best_val_fid = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            loss_val = 0.0
            grads: List[np.ndarray] = [np.zeros_like(p) for p in self.params]
            for inp, tgt in samples:
                pred = self._circuit(inp, self.params)
                fid = state_fidelity(pred, tgt)
                loss_val += (1.0 - fid)
                # Gradient of loss w.r.t parameters
                grad = qml.gradients.param_shift(self._circuit)(inp, self.params)
                grads = [g + dg for g, dg in zip(grads, grad)]

            loss_val /= len(samples)
            grads = [g / len(samples) for g in grads]

            # Parameter update
            self.params = [p - lr * g for p, g in zip(self.params, grads)]
            history.append(loss_val)

            # Early‑stopping based on validation fidelity
            if val_samples is not None:
                val_fid = np.mean(
                    [
                        state_fidelity(self._circuit(inp, self.params), tgt)
                        for inp, tgt in val_samples
                    ]
                )
                if val_fid > best_val_fid:
                    best_val_fid = val_fid
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return self.params, history


__all__ = [
    "GraphQNNGen444QML",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
