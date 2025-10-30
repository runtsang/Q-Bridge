"""Hybrid regression model with classical and quantum components.

The module defines a single ``QuantumRegressionFusion`` class that can be instantiated
with either a purely classical architecture or a hybrid architecture that
includes a variational quantum circuit.  The quantum circuit is built on top of
``torchquantum`` and is wrapped in a ``torch.nn.Module`` so it can be trained
together with the classical layers.  A fidelity‑based graph of the hidden
states is produced via the ``fidelity_adjacency`` helper, which reuses the
functions from the original GraphQNN seed.  The classical side is a small
feed‑forward net that can be used as a baseline or as a head for the quantum
features.

The design unifies the strengths of the two reference pairs:
* The dataset generation and regression head from the first pair.
* The random‑layer encoder, state‑fidelity graph, and graph‑based analysis
  from the second pair.

The class can be used in a single training loop, e.g.:
```
model = QuantumRegressionFusion(num_features=10, num_wires=5, hybrid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for X, y in loader:
    loss = model(X, y)
    loss.backward()
    optimizer.step()
```
"""

from __future__ import annotations

import itertools
import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.quantum as tq  # type: ignore  # torchquantum
from typing import Iterable, List, Tuple

# --------------------------------------------------------------------------- #
# Dataset utilities – identical to the first seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of states |ψ(θ,φ)⟩ = cosθ|0..0⟩ + e^{iφ} sinθ|1..1⟩.

    The labels are y = sin(2θ) cosφ, matching the reference implementation.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset that yields a dict with ``states`` and ``target``."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Fidelity helpers – from GraphQNN
# --------------------------------------------------------------------------- #
def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the absolute squared overlap between two real vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(states: List[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Construct a weighted graph from state fidelities.

    Nodes are the indices of the batch; edges are weighted by the fidelity
    between the hidden states.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# Quantum encoder – from the QML seed
# --------------------------------------------------------------------------- #
class QEncoder(tq.QuantumModule):
    """A trainable variational encoder that maps classical features to a
    superposition state.  It uses a random layer followed by a trainable RX/RY
    on each qubit, mirroring the original QLayer design.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)

# --------------------------------------------------------------------------- #
# Hybrid model – combines classical and quantum
# --------------------------------------------------------------------------- #
class QuantumRegressionFusion(nn.Module):
    """Hybrid regression model.

    Parameters
    ----------
    num_features : int
        Number of classical input features (used for the classical head).
    num_wires : int
        Number of qubits used by the quantum encoder.
    hybrid : bool
        If True, the model uses a quantum encoder and a quantum layer.
        If False, the model falls back to a purely classical feed‑forward net.
    """

    def __init__(self, num_features: int, num_wires: int, hybrid: bool = True):
        super().__init__()
        self.num_features = num_features
        self.num_wires = num_wires
        self.hybrid = hybrid

        # Classical head – identical to the original QNN
        self.classical_head = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        if self.hybrid:
            # Quantum encoder + layer
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
            )
            self.q_layer = QEncoder(num_wires)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.quantum_head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        The method returns the regression loss when ``target`` is supplied,
        otherwise it returns the predictions.
        """
        # --- Classical prediction ------------------------------------------------
        classical_pred = self.classical_head(state_batch)

        if not self.hybrid:
            loss = None
            if target is not None:
                loss = nn.functional.mse_loss(classical_pred, target)
                return loss
            return classical_pred.squeeze(-1)

        # --- Quantum part ---------------------------------------------------------
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.num_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        # Encode classical features into a quantum state
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)

        # Feature extraction – measurement
        features = self.measure(qdev)  # shape (bsz, num_wires)
        quantum_pred = self.quantum_head(features)

        # Combine predictions
        pred = 0.5 * (classical_pred + quantum_pred)

        # --- Loss computation ----------------------------------------------------
        loss = None
        if target is not None:
            loss = nn.functional.mse_loss(pred, target)
            return loss
        return pred.squeeze(-1)

    # ----------------------------------------------------------------------- #
    # Graph analysis utilities
    # ----------------------------------------------------------------------- #
    def build_graph(self, states: Iterable[torch.Tensor], threshold: float,
                    *, secondary: float | None = None,
                    secondary_weight: float = 0.5) -> nx.Graph:
        """Construct a graph from a batch of intermediate states.

        The ``states`` argument is expected to be an iterable of tensors
        (e.g. the hidden activations of the quantum layer).  The graph is
        returned as a networkx.Graph.
        """
        return fidelity_adjacency(
            list(states),
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
        )

__all__ = ["QuantumRegressionFusion", "RegressionDataset", "generate_superposition_data"]
