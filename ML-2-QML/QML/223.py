"""
Hybrid graph‑based QNN built with PennyLane.

The module mirrors the public API of the classical version but
replaces all tensor operations with PennyLane QNodes.  It demonstrates
how to build a variational circuit that mimics a feed‑forward
architecture, how to compute a fidelity‑based adjacency graph from
the intermediate quantum states, and how to jointly optimise the
output energy with a fidelity regulariser.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, List, Sequence

import networkx as nx
import pennylane as qml
import pennylane.numpy as pnp
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _state_fidelity(a: qml.QNode, b: qml.QNode) -> float:
    """Return the squared overlap between two QNodes' output states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def _tensor_fid(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the (unnormalised) squared overlap between two tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b).item() ** 2)


# --------------------------------------------------------------------------- #
# GraphQNNGenQ device setup
# --------------------------------------------------------------------------- #
class GraphQNNGenQ:
    """
    Quantum‑only version of GraphQNNGen that uses a PennyLane device
    and a variational circuit.  The API is intentionally identical to
    the classical implementation to ease comparison.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        dev_name: str = "default.qubit",
        wires: int | None = None,
        seed: int | None = None,
        use_gc: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        qnn_arch : Sequence[int]
            Architecture of the network, e.g. [4, 8, 2].
        dev_name : str, optional
            PennyLane device name.
        wires : int | None, optional
            Number of qubits to run on.  If None, defaults to the
            last layer size.
        seed : int | None, optional
            Seed for reproducibility.
        use_gc : bool, optional
            If True, a simple graph‑convolutional layer is added
            after the last hidden layer.
        """
        self.qnn_arch = list(qnn_arch)
        self.dev_name = dev_name
        self.wires = wires or self.qnn_arch[-1]
        self.use_gc = use_gc

        if seed is not None:
            pnp.random.seed(seed)
            torch.manual_seed(seed)

        self._build_circuit()

    # --------------------------------------------------------------------- #
    # Circuit construction
    # --------------------------------------------------------------------- #
    def _build_circuit(self) -> None:
        dev = qml.device(self.dev_name, wires=self.wires)

        @qml.qnode(dev, interface="torch")
        def circuit(x: torch.Tensor):
            # Encode the input as a computational‑basis state on the first wire.
            for i, val in enumerate(x):
                if val > 0.5:
                    qml.PauliX(i)
            # Layered parametric rotations and entangling gates
            for layer in range(1, len(self.qnn_arch)):
                for w in range(self.qnn_arch[layer]):
                    qml.RY(0.1 * layer, wires=w)
                if layer < len(self.qnn_arch) - 1:
                    qml.CNOT(0, 1)  # simple entangler
            # Measure expectation value of Z on the last qubit
            return qml.expval(qml.PauliZ(self.wires - 1))

        self.circuit = circuit
        # Parameters are stored as a single learnable tensor
        self.params = torch.nn.Parameter(torch.randn(self.wires))

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass and return the output."""
        return self.circuit(x)

    # --------------------------------------------------------------------- #
    # Training helper
    # --------------------------------------------------------------------- #
    def train_step(
        self,
        batch: torch.Tensor,
        target: torch.Tensor,
        lr: float = 1e-3,
        fidelity_weight: float = 0.1,
    ) -> float:
        """
        Perform a single optimisation step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch (N, D_in).
        target : torch.Tensor
            Target output (N, D_out).
        lr : float, optional
            Learning rate.
        fidelity_weight : float, optional
            Weight of the fidelity regulariser.
        """
        optim = torch.optim.Adam([self.params], lr=lr)

        def loss_fn(pred, target):
            mse = F.mse_loss(pred, target)
            reg = 0.0
            # Fidelity regulariser between neighbouring samples
            for i, j in itertools.combinations(range(batch.shape[0]), 2):
                out_i = self.circuit(batch[i])
                out_j = self.circuit(batch[j])
                reg += _tensor_fid(out_i, out_j)
            reg /= (batch.shape[0] * (batch.shape[0] - 1))
            return mse + fidelity_weight * reg

        pred = self.forward(batch)
        loss = loss_fn(pred, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item()

    # --------------------------------------------------------------------- #
    # Graph utilities
    # --------------------------------------------------------------------- #
    def build_fidelity_graph(self, states: Sequence[torch.Tensor]) -> nx.Graph:
        """Construct a weighted graph from quantum states."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, a_i), (j, a_j) in itertools.combinations(enumerate(states), 2):
            fid = _tensor_fid(a_i, a_j)
            if fid > 0.5:
                G.add_edge(i, j, weight=1.0)
            elif fid > 0.2:
                G.add_edge(i, j, weight=0.5)
        return G
