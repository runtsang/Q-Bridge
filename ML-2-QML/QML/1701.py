from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import pennylane as qml
import pennylane.numpy as pnp
import networkx as nx
import numpy as np

class SharedGraphQNN:
    """Quantum graph neural network using PennyLane.

    Parameters
    ----------
    architecture : Sequence[int]
        Number of qubits per layer. The first element is the input size.
    device : str, optional
        A PennyLane device name, e.g. ``default.qubit``.
    """

    def __init__(self, architecture: Sequence[int], device: str = "default.qubit"):
        self.architecture = list(architecture)
        self.num_qubits = max(self.architecture)
        self.dev = qml.device(device, wires=self.num_qubits)
        self.params = self._init_params()
        self._build_circuit()

    def _init_params(self) -> List[np.ndarray]:
        # One parameter per qubit per layer
        return [np.random.randn(n) for n in self.architecture]

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: List[float], params: List[np.ndarray]):
            # encode inputs as rotations
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)
            # apply layer rotations and a simple nearest‑neighbour entangling pattern
            for layer_params, n in zip(params, self.architecture):
                for i in range(n):
                    qml.RZ(layer_params[i], wires=i)
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()
        self.circuit = circuit

    def forward(self, inputs: List[float]) -> List[np.ndarray]:
        """Return the state after the full circuit."""
        return [np.array(inputs), self.circuit(inputs, self.params)]

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Squared overlap for pure states."""
        return float(abs(np.vdot(a, b)) ** 2)

    def fidelity_adjacency(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= threshold:
                g.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                g.add_edge(i, j, weight=secondary_weight)
        return g

    def random_training_data(self, samples: int) -> List[Tuple[List[float], np.ndarray]]:
        data: List[Tuple[List[float], np.ndarray]] = []
        for _ in range(samples):
            inputs = np.random.uniform(-np.pi, np.pi, self.num_qubits).tolist()
            target = self.circuit(inputs, self.params)
            data.append((inputs, target))
        return data

    def train(
        self,
        dataset: List[Tuple[List[float], np.ndarray]],
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        opt = qml.GradientDescentOptimizer(lr)
        for _ in range(epochs):
            for inputs, target in dataset:
                def loss(params):
                    pred = self.circuit(inputs, params)
                    return 1 - self.state_fidelity(pred, target)
                self.params = opt.step(loss, self.params)
