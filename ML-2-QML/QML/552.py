"""Quantum graph neural network implementation using PennyLane.

The ``GraphQNN`` class below mirrors the API of its classical
counterpart but operates on quantum states.  Each layer is a
parameter‑shaped variational unitary that acts on a register
containing the input state and an ancilla of the same size as the
output dimension.  Training is performed with the Adam optimizer
provided by PennyLane, and the model exposes methods for random
network generation, feed‑forward simulation, and fidelity‑based
adjacency graph construction.
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


class GraphQNN:
    """Variational quantum graph network."""

    def __init__(
        self,
        in_qubits: int,
        out_qubits: int,
        num_layers: int = 2,
        device: str = "default.qubit",
    ):
        self.in_qubits = in_qubits
        self.out_qubits = out_qubits
        self.num_layers = num_layers
        self.device = qml.device(device, wires=in_qubits + out_qubits)

        # Parameters: one unitary per layer, each a matrix of size (2**(in+out), 2**(in+out))
        self.params = np.random.randn(
            num_layers, 2 ** (in_qubits + out_qubits), 2 ** (in_qubits + out_qubits)
        )

        @qml.qnode(self.device, interface="autograd")
        def circuit(params, x):
            qml.BasisState(x, wires=range(self.in_qubits))
            for layer in range(self.num_layers):
                qml.ControlledQubitUnitary(params[layer], wires=range(self.in_qubits + self.out_qubits))
            return qml.state()

        self.circuit = circuit

    def feedforward(
        self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]
    ) -> List[List[np.ndarray]]:
        """Simulate the circuit on each input state and return the
        sequence of intermediate states."""
        stored = []
        for inp, _ in samples:
            state = inp
            layerwise = [state]
            for layer in range(self.num_layers):
                state = self.circuit(self.params, state)
                layerwise.append(state)
            stored.append(layerwise)
        return stored

    @staticmethod
    def random_training_data(num_qubits: int, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate random input states and target states via a random unitary."""
        # Random unitary via QR decomposition
        mat = np.random.randn(2 ** num_qubits, 2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits, 2 ** num_qubits)
        q, _ = np.linalg.qr(mat)
        target = q
        dataset = []
        for _ in range(samples):
            inp = np.random.randn(2 ** num_qubits, 1)
            inp /= np.linalg.norm(inp)
            out = target @ inp
            dataset.append((inp, out))
        return dataset

    @staticmethod
    def fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Squared overlap of pure states."""
        return float(np.abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Iterable[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        states = list(states)
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def train(
        self,
        dataset: List[Tuple[np.ndarray, np.ndarray]],
        epochs: int = 20,
        lr: float = 0.01,
    ):
        """Train the variational parameters to approximate the target unitary."""
        opt = qml.AdamOptimizer(lr)
        params = self.params
        for _ in range(epochs):
            for inp, out in dataset:
                loss = np.mean((self.circuit(params, inp) - out) ** 2)
                params = opt.step(loss, params)
        self.params = params
