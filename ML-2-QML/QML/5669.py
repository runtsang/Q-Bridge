"""GraphQNN__gen047.py – Quantum side of the hybrid pipeline.

This module implements a variational quantum neural network that mirrors the
classical interface.  The network is built from a sequence of
parameterised rotation blocks followed by a chain of CNOTs.  A
``qml.QNode`` with ``interface='torch'`` is used so that the
output state is a PyTorch tensor and can be fed into the same
fidelity utilities as the classical model.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import torch

Tensor = torch.Tensor


def _random_qubit_state(num_qubits: int) -> Tensor:
    """Generate a random pure state as a complex torch tensor."""
    dim = 2 ** num_qubits
    vec = torch.randn(dim, dtype=torch.complex128)
    vec /= torch.norm(vec)
    return vec


def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Generate a random unitary matrix as a complex torch tensor."""
    dim = 2 ** num_qubits
    mat = torch.randn(dim, dim, dtype=torch.complex128)
    # QR decomposition to enforce unitarity
    q, r = torch.linalg.qr(mat)
    d = torch.diag(r)
    ph = d / torch.abs(d)
    return q @ torch.diag(ph)


def random_training_data(unitary: Tensor, samples: int):
    """Generate training pairs (state, U*state)."""
    dataset = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.shape[0].bit_length() - 1)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational QNN and a training set."""
    # Target unitary for the final layer
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # Initialise parameters: for each layer and each output we keep a
    # 3‑parameter rotation per qubit (theta, phi, lam).
    params: List[List[Tensor]] = []
    for in_, out in zip(qnn_arch[:-1], qnn_arch[1:]):
        layer_params = []
        for _ in range(out):
            num_qubits = in_ + 1
            # shape: (num_qubits, 3)
            param = torch.randn(num_qubits, 3, dtype=torch.float64)
            layer_params.append(param)
        params.append(layer_params)

    return list(qnn_arch), params, training_data, target_unitary


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two pure states."""
    return float(abs(torch.vdot(a, b).item()) ** 2)


def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Main class – a variational quantum neural network
# --------------------------------------------------------------------------- #
class GraphQNNGen047:
    """A variational QNN that exposes the same interface as its classical counterpart."""

    def __init__(self, arch: Sequence[int], dev: qml.Device | None = None):
        self.arch = list(arch)
        self.dev = dev or qml.device("default.qubit", wires=max(arch))
        # parameters will be initialised by ``random_network`` or can be set manually
        self.params: List[List[Tensor]] | None = None

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Return a random architecture, parameters, training data and a target unitary."""
        return random_network(arch, samples)

    def set_params(self, params: List[List[Tensor]]):
        """Set the trainable parameters of the network."""
        self.params = params

    def _circuit(self, *flat_params: Tensor, sample: Tensor) -> Tensor:
        """Variational circuit that maps the input state to an output state."""
        # Initialise the state
        qml.QubitStateVector(sample, wires=range(self.dev.num_wires))
        # Flatten the parameters back into the original structure
        for layer_idx, layer_params in enumerate(self.params):
            in_wires = list(range(self.arch[layer_idx]))
            for out_idx, out_params in enumerate(layer_params):
                out_wire = self.arch[layer_idx] + out_idx
                wires = in_wires + [out_wire]
                # Apply rotations
                for qubit_idx, (theta, phi, lam) in enumerate(torch.unbind(out_params, dim=0)):
                    qml.Rot(theta, phi, lam, wires=wires[qubit_idx])
                # Entangle with a simple linear CNOT chain
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
        return qml.state()

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        """Return the state after each layer for every sample."""
        if self.params is None:
            raise RuntimeError("Network parameters have not been initialised.")
        stored: List[List[Tensor]] = []
        for sample, _ in samples:
            flat_params = [p for layer in self.params for p in layer]
            qnode = qml.QNode(
                lambda *p: self._circuit(*p, sample=sample),
                self.dev,
                interface="torch",
            )
            state = qnode(*flat_params)
            stored.append([state])
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Return the squared overlap between two pure states."""
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted graph from state fidelities."""
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                  secondary_weight=secondary_weight)


__all__ = [
    "GraphQNNGen047",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
