"""GraphQNNGen354: Quantum graph neural network using Pennylane.

The class mirrors the seed API while adding a variational circuit per
layer, a hybrid training loop, and a fidelity‑based adjacency graph.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Sequence

import itertools

import pennylane as qml
import pennylane.numpy as np
import networkx as nx

Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix using Pennylane."""
    dim = 2 ** num_qubits
    return qml.utils.random_unitary(dim)


def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Generate a random pure state vector."""
    dim = 2 ** num_qubits
    state = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
    state /= np.linalg.norm(state)
    return state


def random_training_data(
    unitary: np.ndarray, samples: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create (input, target) pairs by applying ``unitary`` to random states."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(
    arch: Sequence[int], samples: int
) -> Tuple[Sequence[int], List[List[np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Generate a random target unitary and a set of variational parameters."""
    target_unitary = _random_qubit_unitary(arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # Parameters for each layer: 3 angles per qubit
    params: List[List[np.ndarray]] = []
    for layer in range(1, len(arch)):
        num_qubits = arch[layer - 1]
        layer_params = np.random.normal(size=(num_qubits, 3))
        params.append(layer_params.tolist())
    return arch, params, training_data, target_unitary


# ----------------------------------------------------------------------
# Variational circuit helpers
# ----------------------------------------------------------------------
def _build_circuit(num_qubits: int):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit(params: np.ndarray, input_state: np.ndarray) -> np.ndarray:
        qml.StatePrep(input_state, wires=range(num_qubits))
        for q in range(num_qubits):
            qml.RX(params[q, 0], wires=q)
            qml.RY(params[q, 1], wires=q)
            qml.RZ(params[q, 2], wires=q)
        for q in range(num_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        return qml.state()

    return circuit


class GraphQNNGen354:
    """Quantum graph‑neural‑network with variational layers."""

    def __init__(self, arch: Sequence[int]):
        self.arch: Sequence[int] = tuple(arch)
        self.num_qubits: int = arch[0]
        self.circuits = [_build_circuit(self.num_qubits) for _ in range(len(arch) - 1)]
        # Flatten all parameters into a list of numpy arrays
        self.params: List[np.ndarray] = [
            np.random.normal(size=(self.num_qubits, 3)) for _ in range(len(arch) - 1)
        ]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, input_state: np.ndarray) -> List[np.ndarray]:
        """Apply all layers sequentially, returning intermediate states."""
        states: List[np.ndarray] = [input_state]
        state = input_state
        for circuit, params in zip(self.circuits, self.params):
            state = circuit(params, state)
            states.append(state)
        return states

    # ------------------------------------------------------------------
    # Compatibility wrapper
    # ------------------------------------------------------------------
    def feedforward(
        self,
        samples: Iterable[Tuple[np.ndarray, np.ndarray]],
    ) -> List[List[np.ndarray]]:
        """Return layerwise states for each sample."""
        stored: List[List[np.ndarray]] = []
        for state, _ in samples:
            stored.append(self.forward(state))
        return stored

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        dataset: Iterable[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 50,
        lr: float = 0.01,
    ) -> List[float]:
        """Hybrid Adam training of variational parameters."""
        optimizer = qml.optimize.AdamOptimizer(stepsize=lr)
        losses: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for input_state, target_state in dataset:
                # Define cost for the current sample
                def cost(p_list):
                    state = input_state
                    for circ, p in zip(self.circuits, p_list):
                        state = circ(p, state)
                    return 1 - GraphQNNGen354.state_fidelity(state, target_state)

                # Compute gradients for each layer sequentially
                for i, circ in enumerate(self.circuits):
                    def cost_layer(p):
                        state = input_state
                        for j, c in enumerate(self.circuits):
                            if j == i:
                                state = c(p, state)
                            else:
                                state = c(self.params[j], state)
                        return 1 - GraphQNNGen354.state_fidelity(state, target_state)

                    grad = qml.grad(cost_layer)(self.params[i])
                    self.params[i] = optimizer.step(grad, self.params[i])

                # Loss after full forward pass
                state = input_state
                for circ, p in zip(self.circuits, self.params):
                    state = circ(p, state)
                epoch_loss += 1 - GraphQNNGen354.state_fidelity(state, target_state)

            epoch_loss /= len(dataset)
            losses.append(epoch_loss)
            print(f"Epoch {epoch}, loss={epoch_loss}")
        return losses

    # ------------------------------------------------------------------
    # Fidelity utilities
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Squared magnitude of inner product for pure states."""
        return abs(np.vdot(a, b)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen354.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph
