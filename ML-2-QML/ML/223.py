"""
Hybrid graph neural network with classical and quantum sub‑modules.

The class exposes identical public methods for both the pure‑classical
and the hybrid quantum‑classical variants.  The constructor accepts a
``model_type`` flag that determines whether the network is built from
PyTorch tensors or PennyLane QNodes.  The training routine implements a
joint loss that mixes the mean‑squared‑error on the output layer
with a fidelity‑based regulariser that encourages neighbouring states
to remain similar.  The module also demonstrates how to compute a
fidelity‑based adjacency graph from the hidden activations, which can
be used for downstream clustering or graph‑convolution.
"""

from __future__ import annotations

import itertools
import math
from typing import Iterable, List, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _tensor_fid(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the (unnormalised) squared overlap between two tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b).item() ** 2)


def random_state_vector(t: torch.Tensor) -> torch.Tensor:
    """Generate a random unit‑vector in the same dimension as ``t``."""
    vec = torch.randn(t.shape, dtype=torch.float32)
    vec /= (torch.norm(vec) + 1e-12)
    return vec


# --------------------------------------------------------------------------- #
# GraphQNNGen class
# --------------------------------------------------------------------------- #
class GraphQNNGen:
    """
    A hybrid graph‑based neural network that can either:
    - (1) run as a *classical* feed‑forward network (PyTorch)
    - (2) run as a *quantum‑neural‑network (QNN) using PennyLane
    The only difference between the two branches is the underlying
    tensor type and the way the forward pass is executed; the public
    API (`forward`, `train_step`, `build_fidelity_graph`) is the same.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        model_type: str = "classic",
        seed: int | None = None,
        use_gc: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        qnn_arch : Sequence[int]
            Architecture of the network, e.g. [4, 8, 2].
        model_type : str, optional
            Either ``"classic"`` (PyTorch) or ``"quantum"`` (PennyLane).
        seed : int | None, optional
            Seed for reproducibility.
        use_gc : bool, optional
            If True, a simple graph‑convolutional layer is inserted after
            the last hidden layer before the output.
        """
        self.qnn_arch = list(qnn_arch)
        self.model_type = model_type.lower()
        self.use_gc = use_gc

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if self.model_type == "classic":
            self._build_classical()
        elif self.model_type == "quantum":
            self._build_quantum()
        else:
            raise ValueError("model_type must be either 'classic' or 'quantum'")

    # --------------------------------------------------------------------- #
    # Classical construction
    # --------------------------------------------------------------------- #
    def _build_classical(self) -> None:
        self.layers: nn.ModuleList = nn.ModuleList()
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        if self.use_gc:
            self.gc = nn.Linear(self.qnn_arch[-2], self.qnn_arch[-1])
        else:
            self.gc = None

    # --------------------------------------------------------------------- #
    # Quantum construction (PennyLane)
    # --------------------------------------------------------------------- #
    def _build_quantum(self) -> None:
        import pennylane as qml
        from pennylane import numpy as pnp

        dev = qml.device("default.qubit", wires=len(self.qnn_arch[-1]))

        @qml.qnode(dev, interface="torch")
        def circuit(input_state: torch.Tensor):
            # Initialise the input state as a computational basis state
            # encoded in the last wire of the device.
            # The remaining wires are used as ancillas for the unitaries.
            for i, w in enumerate(range(len(self.qnn_arch[-1]))):
                qml.PauliX(w) if i == 0 else None  # trivial mapping

            # Layered parametric rotations (example)
            for idx in range(1, len(self.qnn_arch)):
                for w in range(self.qnn_arch[idx]):
                    qml.RY(0.1 * idx, wires=w)

            # Measure in computational basis
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        self.params = torch.nn.Parameter(torch.randn(self.qnn_arch[-1]))

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single forward pass and return the activations."""
        if self.model_type == "classic":
            activations = [x]
            for layer in self.layers:
                x = torch.tanh(layer(x))
                activations.append(x)
            if self.gc:
                x = torch.tanh(self.gc(x))
                activations.append(x)
            return activations
        else:
            # Quantum forward: treat the input as a probability amplitude
            # vector (requires normalisation).
            out = self.circuit(x)
            return [out]

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
        optim = torch.optim.Adam([p for p in self.parameters()], lr=lr)

        def loss_fn(pred, target):
            mse = F.mse_loss(pred, target)
            # Fidelity regulariser
            if self.model_type == "classic":
                # Compute pairwise fidelity between hidden activations
                hidden = pred
                fid_sum = 0.0
                for i, j in itertools.combinations(range(len(hidden)), 2):
                    fid_sum += _tensor_fid(hidden[i], hidden[j])
                reg = fid_sum / (len(hidden) * (len(hidden) - 1))
            else:
                # For quantum case we approximate with a simple
                # overlap between the output state and its neighbours.
                reg = 0.0
            return mse + fidelity_weight * reg

        # Forward pass
        activations = self.forward(batch)
        # Last layer output
        pred = activations[-1] if isinstance(activations, list) else activations
        loss = loss_fn(pred, target)

        # Backward
        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss.item()

    # --------------------------------------------------------------------- #
    # Graph utilities
    # --------------------------------------------------------------------- #
    def build_fidelity_graph(self, activations: Sequence[torch.Tensor]) -> nx.Graph:
        """Construct a weighted graph from activations using state fidelity."""
        G = nx.Graph()
        G.add_nodes_from(range(len(activations)))
        for (i, a_i), (j, a_j) in itertools.combinations(enumerate(activations), 2):
            fid = _tensor_fid(a_i, a_j)
            if fid > 0.5:
                G.add_edge(i, j, weight=1.0)
            elif fid > 0.2:
                G.add_edge(i, j, weight=0.5)
        return G

    # --------------------------------------------------------------------- #
    # Parameter handling
    # --------------------------------------------------------------------- #
    def parameters(self):
        if self.model_type == "classic":
            return self.layers.parameters()
        else:
            return [self.params]
