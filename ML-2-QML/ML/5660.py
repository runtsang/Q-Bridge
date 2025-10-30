"""Hybrid regression model that fuses dense neural processing with a quantum module.

The module implements a two‑stage pipeline:
1. Classical encoder (dense layers) that extracts low‑level features.
2. Quantum variational layer that further transforms the features via a unitary
   parametrised by trainable parameters.
3. Attention‑style aggregation that uses a fidelity‑based graph over the batch
   to weight the quantum outputs before the final linear head.

The classical and quantum parts are trained jointly; the fidelity graph is
computed on the fly from the batch states and is used as a learnable
attention mask.  The code is fully importable and can be run on CPU or GPU.
"""

from __future__ import annotations

import itertools
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Data generation – synthetic superposition data
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a batch of synthetic states and targets."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields ``states`` and ``target`` tensors."""
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
# Fidelity graph utilities – borrowed from the second seed
# --------------------------------------------------------------------------- #
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return squared cosine similarity between two feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_graph(
    states: torch.Tensor,
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a weighted graph from pairwise fidelities."""
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
# Quantum sub‑module – variational circuit
# --------------------------------------------------------------------------- #
class _QuantumLayer(tq.QuantumModule):
    """Quantum variational circuit that accepts the classical feature vector
    as a state and outputs a feature vector of size ``num_wires``."""
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder that maps a real vector to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Parametric random layer
        self.var_layer = tq.RandomLayer(
            n_ops=30,
            wires=range(num_wires),
            has_params=True,
            trainable=True,
        )
        # Measurement of all qubits in Pauli‑Z
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Return a batch of quantum features."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.var_layer(qdev)
        features = self.measure(qdev)
        return features  # shape (bsz, n_wires)

# --------------------------------------------------------------------------- #
# Hybrid model definition
# --------------------------------------------------------------------------- #
class UnifiedQuantumRegressor(nn.Module):
    """Hybrid classical‑quantum regression model."""
    def __init__(
        self,
        num_features: int,
        num_wires: int,
        hidden_dim: int = 32,
        fidelity_threshold: float = 0.8,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_layer = _QuantumLayer(num_wires)
        self.head = nn.Linear(num_wires, 1)
        self.fidelity_threshold = fidelity_threshold

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that mixes classical and quantum features
        and aggregates them using a fidelity‑based attention graph.
        """
        # Classical encoding
        enc = self.encoder(state_batch)  # (bsz, hidden_dim)

        # Quantum pass – use the encoded features as the state
        q_out = self.q_layer(enc)  # (bsz, num_wires)

        # Build fidelity graph over quantum outputs
        graph = fidelity_graph(q_out, self.fidelity_threshold)

        # Attention: for each node, weight its features by the mean weight of its neighbours
        attn = torch.zeros_like(q_out)
        for i in graph.nodes:
            neighbors = list(graph.neighbors(i))
            if neighbors:
                weights = torch.tensor(
                    [graph[i][j]["weight"] for j in neighbors],
                    device=q_out.device,
                )
                attn[i] = q_out[i] * weights.mean()
            else:
                attn[i] = q_out[i]

        # Aggregate over the batch (sum weighted features)
        agg = attn.sum(dim=0, keepdim=True)  # (1, num_wires)

        # Final linear head
        out = self.head(agg).squeeze(-1)  # (1,)
        return out

__all__ = [
    "UnifiedQuantumRegressor",
    "RegressionDataset",
    "generate_superposition_data",
]
