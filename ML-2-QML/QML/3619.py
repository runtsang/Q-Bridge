"""Hybrid graph neural network for quantum processing.

Adopts the same public API as the classical module but replaces
linear layers with parameterised quantum circuits.  Uses PennyLane
to construct random unitaries, propagate quantum states and evaluate
an overlap‑based quantum kernel.  The class is deliberately
drop‑in compatible with :class:`GraphQNNGen` from the classical
module, enabling seamless experimentation across regimes.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import pennylane as qml
import numpy as np
import networkx as nx

State = np.ndarray

class GraphQNNGen:
    """
    Hybrid graph neural network with quantum layers.

    Parameters
    ----------
    arch : Sequence[int]
        Widths of quantum layers.
    gamma : float, default 1.0
        RBF width retained for back‑compatibility.
    num_samples : int, default 100
        Number of synthetic training samples.
    device_name : str, default "default.qubit"
        PennyLane backend.
    """

    def __init__(
        self,
        arch: Sequence[int],
        gamma: float = 1.0,
        num_samples: int = 100,
        device_name: str = "default.qubit",
    ) -> None:
        self.arch = list(arch)
        self.gamma = gamma
        self.num_samples = num_samples
        self.dev = qml.device(device_name, wires=max(arch))
        self.weights: List[List[np.ndarray]] = []

    # --------------------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------------------

    @staticmethod
    def _random_unitary(num_qubits: int) -> np.ndarray:
        dim = 2 ** num_qubits
        random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, r = np.linalg.qr(random_matrix)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        return q @ np.diag(ph)

    def _partial_trace(self, state: State, keep: Sequence[int]) -> State:
        dim = state.shape[0]
        n_qubits = int(np.log2(dim))
        state_reshaped = state.reshape([2] * n_qubits + [1])
        for wire in sorted(set(range(n_qubits)) - set(keep), reverse=True):
            state_reshaped = np.trace(state_reshaped, axis1=wire, axis2=wire + n_qubits)
        return state_reshaped.reshape(-1, 1)

    # --------------------------------------------------------------------
    # GraphQNN utilities
    # --------------------------------------------------------------------

    def random_network(self) -> Tuple[List[int], List[List[np.ndarray]], List[Tuple[State, State]], State]:
        unitaries: List[List[np.ndarray]] = [[]]
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[np.ndarray] = []
            for _ in range(num_outputs):
                op = self._random_unitary(num_inputs + 1)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        target_unitary = self._random_unitary(self.arch[-1])
        training_data = self.random_training_data(target_unitary, self.num_samples)
        self.weights = unitaries
        return self.arch, unitaries, training_data, target_unitary

    def random_training_data(self, unitary: np.ndarray, samples: int) -> List[Tuple[State, State]]:
        dataset: List[Tuple[State, State]] = []
        dim = unitary.shape[0]
        for _ in range(samples):
            state = np.random.randn(dim, 1) + 1j * np.random.randn(dim, 1)
            state /= np.linalg.norm(state)
            out = unitary @ state
            dataset.append((state, out))
        return dataset

    def _layer_channel(
        self,
        layer: int,
        input_state: State,
    ) -> State:
        num_inputs = self.arch[layer - 1]
        unitary = self.weights[layer][0]
        for gate in self.weights[layer][1:]:
            unitary = gate @ unitary
        zeros = np.zeros((2 ** (self.arch[layer] - 1), 1), dtype=complex)
        full_state = np.kron(input_state, zeros)
        out_state = unitary @ full_state
        keep = list(range(num_inputs, num_inputs + self.arch[layer]))
        return self._partial_trace(out_state, keep)

    def feedforward(
        self,
        samples: Iterable[Tuple[State, State]],
    ) -> List[List[State]]:
        stored: List[List[State]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(self.arch)):
                current = self._layer_channel(layer, current)
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    def state_fidelity(self, a: State, b: State) -> float:
        return np.abs(np.vdot(a, b)) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[State],
        threshold: float,
        *,
        secondary: Optional[float] = None,
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

    # --------------------------------------------------------------------
    # Quantum kernel utilities
    # --------------------------------------------------------------------

    def quantum_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        wires = len(x)
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev, diff_method=None)
        def encode(data):
            for i, val in enumerate(data):
                qml.RY(val, wires=i)
            return qml.state()

        psi_x = encode(x)
        psi_y = encode(y)
        return np.abs(np.vdot(psi_x, psi_y)) ** 2

    def kernel_matrix(
        self,
        a: Sequence[State],
        b: Sequence[State],
    ) -> np.ndarray:
        return np.array(
            [[self.quantum_kernel(x.squeeze(), y.squeeze()) for y in b] for x in a]
        )

__all__ = [
    "GraphQNNGen",
]
