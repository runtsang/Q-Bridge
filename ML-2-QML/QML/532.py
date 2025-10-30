"""Hybrid Graph Neural Network with quantum variational circuit.

The module extends the original quantum‑only GraphQNN by adding a
variational circuit built with Pennylane and a classical linear head.
The class ``HybridGraphQNN`` can be trained on a CPU‑only simulator
(``default.qubit``) with autograd support.  The public API mirrors the
seed code: ``random_network``, ``random_training_data``, ``feedforward``,
``state_fidelity`` and ``fidelity_adjacency`` remain available as
standalone functions.

The training loop updates the circuit parameters and the linear head
using mean‑squared error loss.  When ``train_circuit=False`` the
circuit is frozen and only the head is trained.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as pnp  # for deterministic random numbers
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Helper functions (original seed logic with small extensions)
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Return a random unitary matrix as a torch tensor on the CPU."""
    dim = 2 ** num_qubits
    random_matrix = pnp.random.randn(dim, dim) + 1j * pnp.random.randn(dim, dim)
    unitary, _ = pnp.linalg.qr(random_matrix)
    return torch.tensor(unitary, dtype=torch.complex64)

def random_training_data(
    unitary: Tensor, samples: int
) -> List[Tuple[Tensor, Tensor]]:
    """Generate ``samples`` input states and their images under ``unitary``."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    num_qubits = int(math.log2(unitary.shape[0]))
    for _ in range(samples):
        state = pnp.random.randn(unitary.shape[0]) + 1j * pnp.random.randn(unitary.shape[0])
        state = state / pnp.linalg.norm(state)
        target = unitary @ state
        dataset.append((torch.tensor(state, dtype=torch.complex64),
                        torch.tensor(target, dtype=torch.complex64)))
    return dataset

def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Generate a random variational circuit and training data."""
    # Number of qubits equals the width of the last layer
    num_qubits = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_qubits)
    training_data = random_training_data(target_unitary, samples)
    # Parameters for a StronglyEntanglingLayers circuit
    num_layers = len(qnn_arch) - 1
    params = torch.randn(num_layers, num_qubits, 3, dtype=torch.float32)
    return list(qnn_arch), [params], training_data, target_unitary

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between pure states a and b."""
    return abs((a.conj().t() @ b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[Tensor],
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

# --------------------------------------------------------------------------- #
# HybridGraphQNN class
# --------------------------------------------------------------------------- #

class HybridGraphQNN:
    """Hybrid quantum‑classical graph neural network.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the variational circuit: a list of layer widths.
    dev : str, optional
        Pennylane device name.  Defaults to ``default.qubit``.
    head_dim : int, optional
        Output dimension of the classical head.  Defaults to 1.
    train_circuit : bool, optional
        If ``True`` the circuit parameters are updated during training.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        dev: str = "default.qubit",
        head_dim: int = 1,
        train_circuit: bool = False,
    ) -> None:
        self.arch = list(qnn_arch)
        self.dev_name = dev
        self.num_qubits = self.arch[-1]
        self.train_circuit = train_circuit

        # Circuit parameters
        self.num_layers = len(self.arch) - 1
        self.params = nn.Parameter(
            torch.randn(self.num_layers, self.num_qubits, 3, dtype=torch.float32)
        )

        # Classical head: map 2*num_qubits (real+imag) to head_dim
        self.head = nn.Linear(2 * self.num_qubits, head_dim, bias=True).to("cpu")

        # Optimizer placeholder (set in ``fit``)
        self.optimizer: optim.Optimizer | None = None

        # Pennylane device (CPU‑only)
        self.dev = qml.device(self.dev_name, wires=self.num_qubits)

        # QNode: returns the final state vector
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(params: Tensor, input_state: Tensor) -> Tensor:
            # Prepare the input state
            qml.QubitStateVector(input_state, wires=range(self.num_qubits))
            # Apply a strongly entangling layer sequence
            qml.StronglyEntanglingLayers(params, wires=range(self.num_qubits))
            return qml.state()

        self._circuit = circuit

    # ----------------------------------------------------------------------- #
    # Forward pass
    # ----------------------------------------------------------------------- #
    def _forward(self, x: Tensor) -> Tensor:
        """Return the final state vector after the variational circuit."""
        # x is a complex state vector of shape (2**num_qubits,)
        state = self._circuit(self.params, x)
        # Flatten real and imaginary parts for the classical head
        real = state.real.view(-1)
        imag = state.imag.view(-1)
        features = torch.cat([real, imag], dim=-1)
        return self.head(features)

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #
    def predict(self, X: Iterable[Tensor]) -> Tensor:
        """Apply the network to ``X`` and return the head output."""
        self.eval()
        with torch.no_grad():
            preds = [self._forward(x) for x in X]
        return torch.stack(preds)

    def eval(self) -> None:
        """Set the network to evaluation mode."""
        self.head.eval()
        self.params.requires_grad = False

    def train(self) -> None:
        """Set the network to training mode."""
        self.head.train()
        self.params.requires_grad = self.train_circuit

    # ----------------------------------------------------------------------- #
    # Training loop
    # ----------------------------------------------------------------------- #
    def fit(
        self,
        training_data: Iterable[Tuple[Tensor, Tensor]],
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 0.0,
    ) -> None:
        """Train the head (and optionally the circuit) on ``training_data``."""
        self.train()
        # Build optimizer
        params = list(self.head.parameters())
        if self.train_circuit:
            params.append(self.params)
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in training_data:
                self.optimizer.zero_grad()
                pred = self._forward(x)
                loss = loss_fn(pred, y.real.to("cpu"))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss/len(training_data):.4f}")

    # ----------------------------------------------------------------------- #
    # Utility methods (mirroring the seed functions)
    # ----------------------------------------------------------------------- #
    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        """Generate a new random network with the current architecture."""
        return random_network(self.arch, samples)

    def random_training_data(self, unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate training data for a given unitary."""
        return random_training_data(unitary, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Wrap the original fidelity adjacency construction."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def state_fidelity(self, a: Tensor, b: Tensor) -> float:
        """Wrap the original state fidelity computation."""
        return state_fidelity(a, b)

    def __repr__(self) -> str:
        return f"<HybridGraphQNN arch={self.arch} head_dim={self.head.out_features} train_circuit={self.train_circuit}>"

# --------------------------------------------------------------------------- #
# Public exports
# --------------------------------------------------------------------------- #

__all__ = [
    "HybridGraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
