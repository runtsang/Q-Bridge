"""Graph‑based quantum neural network module using PennyLane variational circuits.

The quantum side replaces the raw unitary construction from the seed with a
parameterised variational circuit that can be optimised with automatic
differentiation.  The module still exposes ``feedforward`` and
``fidelity_adjacency`` so that it can be swapped with the classical version
in existing experiments.  A simple graph‑based regulariser is also provided
to match the classical module’s loss function.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import QuantumTape, QNode
from pennylane import default_qubit


Tensor = pnp.ndarray
State = qml.QubitStateVector


@dataclass
class GraphQNNConfig:
    """Configuration for the quantum graph neural network."""
    hidden_layers: List[int] = None
    """Number of qubits per layer (first element is input qubits)."""
    num_shots: int = 1024
    """Number of shots for the circuit simulation."""
    backend: str = "default.qubit"
    """PennyLane backend to use for simulation."""


class GraphQNN:
    """Variational graph quantum neural network."""

    def __init__(self, config: GraphQNNConfig):
        self.config = config
        self.layers: List[QNode] = []
        self._build_circuit()

    def _build_circuit(self):
        """Create a variational circuit for each layer."""
        for i in range(1, len(self.config.hidden_layers)):
            num_qubits = self.config.hidden_layers[i]
            dev = qml.device(self.config.backend, wires=num_qubits, shots=self.config.num_shots)

            @qml.qnode(dev, interface="autograd")
            def circuit(inputs: Tensor, params: Tensor):
                # encode inputs
                for w in range(num_qubits):
                    qml.RX(inputs[w], wires=w)
                # variational block
                for w in range(num_qubits):
                    qml.RZ(params[w], wires=w)
                    qml.CNOT(wires=[w, (w + 1) % num_qubits])
                return qml.state()

            self.layers.append(circuit)

    def feedforward(
        self, samples: Iterable[Tuple[State, State]]
    ) -> List[List[State]]:
        """Run the variational circuit for each layer and collect the state vectors."""
        stored_states: List[List[State]] = []
        for state, _ in samples:
            layerwise: List[State] = [state]
            current_state = state
            for layer in self.layers:
                # We treat the state as a vector of amplitudes and feed it into the circuit.
                # The circuit expects a real vector of shape (num_qubits,)
                # and outputs the final state vector.
                params = pnp.random.randn(current_state.shape[0])
                current_state = layer(current_state, params)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    def fidelity_adjacency(
        self, states: Sequence[State], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
    ) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def state_fidelity(a: State, b: State) -> float:
        """Return the absolute squared overlap between pure states."""
        return abs((a.conj().T @ b)[0, 0]) ** 2

    def loss_with_graph_reg(
        self, predictions: Tensor, targets: Tensor, graph: nx.Graph
    ) -> Tensor:
        """Mean‑squared error plus a graph‑based regularisation term."""
        mse = qml.math.mean((predictions - targets) ** 2)
        reg = 0.0
        if self.config.num_shots > 0 and graph.number_of_edges() > 0:
            for (i, j) in graph.edges():
                reg += qml.math.norm(predictions[i] - predictions[j]) ** 2
            reg = reg / graph.number_of_edges()
            reg *= 0.1  # fixed weight for demonstration
        return mse + reg


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[QNode], List[Tuple[State, State]], State]:
    """Create a random variational circuit network and training data."""
    target_unitary = qml.QubitUnitary(pnp.random.randn(2 ** qnn_arch[-1], 2 ** qnn_arch[-1]), wires=range(qnn_arch[-1]))
    training_data = []
    for _ in range(samples):
        state = pnp.random.randn(2 ** qnn_arch[-1])
        state = state / pnp.linalg.norm(state)
        training_data.append((state, target_unitary @ state))
    return list(qnn_arch), [target_unitary], training_data, target_unitary


def feedforward(
    qnn_arch: Sequence[int], unitaries: Sequence[QNode], samples: Iterable[Tuple[State, State]]
) -> List[List[State]]:
    """Legacy wrapper that keeps the original signature."""
    stored: List[List[State]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = unitaries[layer] @ current_state
            layerwise.append(current_state)
        stored.append(layerwise)
    return stored


__all__ = [
    "GraphQNN",
    "GraphQNNConfig",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
