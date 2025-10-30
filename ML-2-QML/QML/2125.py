import itertools
from typing import Iterable, List, Sequence, Tuple

import pennylane as qml
import pennylane.numpy as pnp
import networkx as nx
import numpy as np

class GraphQNN__gen343:
    # Quantum graph neural network using Pennylane.
    def __init__(self, qnn_arch: Sequence[int], dev: str = "default.qubit", shots: int = 1024):
        self.arch = list(qnn_arch)
        self.dev = qml.device(dev, wires=self.arch[-1], shots=shots)
        self.params = self._init_params()
        # Wrap the internal circuit as a QNode
        self.circuit = qml.QNode(self._circuit, self.dev)

    def _init_params(self) -> List[pnp.ndarray]:
        params: List[pnp.ndarray] = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            # For each output qubit we create a 3‑parameter rotation block per input qubit
            layer_params = pnp.random.randn(out_f, in_f, 3)
            params.append(layer_params)
        return params

    def _circuit(self, x: np.ndarray, params: List[pnp.ndarray]):
        # Feature encoding by RX rotations
        for i, val in enumerate(x):
            qml.RX(val, wires=i)
        # Variational layers
        for layer_params in params:
            for out_idx in range(layer_params.shape[0]):
                for in_idx in range(layer_params.shape[1]):
                    rx, rz, ry = layer_params[out_idx, in_idx]
                    qml.RX(rx, wires=in_idx)
                    qml.RZ(rz, wires=in_idx)
                    qml.RY(ry, wires=in_idx)
            # Entangle adjacent qubits
            for i in range(layer_params.shape[0] - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.state()

    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        stored: List[List[np.ndarray]] = []
        for sample, _ in samples:
            state = self.circuit(sample, self.params)
            stored.append([state])
        return stored

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-12)
        b_norm = b / (np.linalg.norm(b) + 1e-12)
        return float(np.abs(np.vdot(a_norm, b_norm)) ** 2)

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
            fid = GraphQNN__gen343.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        dataset: List[Tuple[np.ndarray, np.ndarray]] = []
        dim = unitary.shape[0]
        for _ in range(samples):
            state = np.random.randn(dim) + 1j * np.random.randn(dim)
            state /= np.linalg.norm(state)
            target = unitary @ state
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        dim = 2 ** qnn_arch[-1]
        rand_mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        target_unitary, _ = np.linalg.qr(rand_mat)
        training_data = GraphQNN__gen343.random_training_data(target_unitary, samples)
        return list(qnn_arch), None, training_data, target_unitary

    def train(
        self,
        dataset: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 100,
        lr: float = 0.01,
        fidelity_reg: float = 0.0,
    ) -> None:
        opt = qml.AdamOptimizer(stepsize=lr)
        params = self.params
        for _ in range(epochs):
            for x, y in dataset:
                def cost(p):
                    pred = self.circuit(x, p)
                    mse = np.mean((pred - y) ** 2)
                    if fidelity_reg > 0.0:
                        fid = self.state_fidelity(pred, y)
                        mse += fidelity_reg * (1.0 - fid)
                    return mse
                params, _ = opt.step_and_cost(cost, params)
        self.params = params
