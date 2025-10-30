"""
HybridGraphQNN: quantum graph neural network with a variational ansatz.

The class implements a variational circuit that mirrors a classical linear architecture,
provides random training data, a fidelity-based adjacency graph, and a simple gradient
descent training loop that optimises the circuit parameters to match target states.
"""

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

class HybridGraphQNN:
    """Quantum hybrid graph neural network with a variational ansatz."""

    def __init__(self, arch: Sequence[int], device: str = "default.qubit", shots: int = 1000):
        self.arch = list(arch)
        self.num_wires = max(arch)
        self.device = qml.device(device, wires=self.num_wires, shots=shots)
        # Parameters: one rotation (RX,RZ) per wire per layer
        self.params = np.random.randn(self.num_wires, 3)

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Prepare input state as computational basis
        for i in range(self.num_wires):
            if x[i] > 0.5:
                qml.PauliX(i)
        # Apply rotations
        for wire in range(self.num_wires):
            qml.RX(params[wire, 0], wires=wire)
            qml.RY(params[wire, 1], wires=wire)
            qml.RZ(params[wire, 2], wires=wire)
        # Entangle all wires
        for i in range(self.num_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

    def forward(self, x: np.ndarray) -> np.ndarray:
        @qml.qnode(self.device)
        def circuit_fn(x, params):
            return self._circuit(x, params)
        return circuit_fn(x, self.params)

    def state_fidelity(self, a: np.ndarray, b: np.ndarray) -> float:
        return abs(np.vdot(a, b)) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def random_training_data(self, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        data = []
        for _ in range(samples):
            x = np.random.randint(0, 2, size=self.num_wires)
            target = self.forward(x)
            data.append((x, target))
        return data

    def train(self, data: List[Tuple[np.ndarray, np.ndarray]], lr: float = 0.01, epochs: int = 100):
        opt = qml.GradientDescentOptimizer(lr)
        for _ in range(epochs):
            for x, target in data:
                params, _ = opt.step_and_cost(
                    lambda p: 1 - self.state_fidelity(self.forward(x), target),
                    self.params,
                )
                self.params = params

__all__ = [
    "HybridGraphQNN",
]
