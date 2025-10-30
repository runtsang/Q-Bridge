"""GraphQNN__gen110 - Classical GNN + hybrid training loop.

The module keeps the original forward‑propagation helpers and fidelity‑based
adjacency construction but adds:

* a graph‑convolutional pre‑processing layer that turns node features into
  embeddings;
* a PyTorch GNN that predicts the parameters of a variational quantum circuit;
* a very small synthetic training loop that can be run on CPU.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import GraphQNN__gen110_qml as qml

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1.  GNN model
# --------------------------------------------------------------------------- #
class GraphNet(nn.Module):
    """
    Simple two‑layer GCN that maps node features to a vector of variational
    circuit parameters.  The final pooling aggregates over all nodes to
    produce a global representation.
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.conv1 = nn.Linear(in_features, hidden_features)
        self.conv2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # Message passing
        h = torch.matmul(adj, x)
        h = self.conv1(h)
        h = F.relu(h)
        h = torch.matmul(adj, h)
        h = self.conv2(h)
        # Global pooling
        return h.mean(dim=0)


# --------------------------------------------------------------------------- #
# 2.  Data generation utilities
# --------------------------------------------------------------------------- #
def generate_synthetic_graphs(
    num_graphs: int,
    num_nodes: int,
    node_features: int,
    num_qubits: int,
) -> List[Dict]:
    """
    Create a list of random graphs together with node features and a target
    unitary that acts on ``num_qubits`` qubits.
    """
    graphs: List[Dict] = []
    for _ in range(num_graphs):
        # Random graph
        G = nx.erdos_renyi_graph(num_nodes, 0.5)
        adj = nx.to_numpy_array(G, dtype=float)
        # Add self‑loops
        adj += np.eye(num_nodes)
        # Node features
        features = np.random.randn(num_nodes, node_features).astype(np.float32)
        # Target unitary for the variational circuit
        target_unitary = qml._random_qubit_unitary(num_qubits)
        graphs.append({
            "adj": adj,
            "features": features,
            "target_unitary": target_unitary,
        })
    return graphs


def random_state_vector(num_qubits: int) -> Tensor:
    """
    Sample a random pure state on ``num_qubits`` qubits, returned as a
    complex torch vector.
    """
    dim = 2 ** num_qubits
    vec = torch.randn(dim, dtype=torch.complex64)
    vec = vec / torch.linalg.norm(vec)
    return vec


# --------------------------------------------------------------------------- #
# 3.  Hybrid training loop
# --------------------------------------------------------------------------- #
def train_gnn_and_qml(
    graphs: List[Dict],
    num_epochs: int = 20,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> GraphNet:
    """
    Train the GNN to predict parameters for a variational circuit that
    approximates the target unitary associated with each graph.
    """
    # Build the GNN
    in_features = graphs[0]["features"].shape[1]
    hidden_features = 32
    out_features = qml.NUM_QUBITS  # number of variational parameters
    gnn = GraphNet(in_features, hidden_features, out_features).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for g in graphs:
            adj = torch.tensor(g["adj"], dtype=torch.float32, device=device)
            features = torch.tensor(g["features"], dtype=torch.float32, device=device)
            # Predict parameters
            params = gnn(features, adj)  # shape [out_features]
            # Sample a handful of input states
            input_states = [random_state_vector(qml.NUM_QUBITS).to(device) for _ in range(4)]
            # Compute target output states via the target unitary
            target_states = [qml._random_qubit_unitary(qml.NUM_QUBITS) * qml._random_qubit_state(qml.NUM_QUBITS) for _ in range(4)]
            # Run the variational circuit with the predicted parameters
            output_states = [qml.apply_variational_circuit(params, ts) for ts in target_states]
            # Compute fidelities
            fidelities: List[Tensor] = []
            for out_state, tgt_state in zip(output_states, target_states):
                # Convert to torch vectors
                out_vec = torch.tensor(out_state.full().flatten(), dtype=torch.complex64, device=device)
                tgt_vec = torch.tensor(tgt_state.full().flatten(), dtype=torch.complex64, device=device)
                fid = torch.abs(torch.dot(out_vec, tgt_vec.conj())) ** 2
                fidelities.append(fid)
            loss = 1 - torch.stack(fidelities).mean()
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1:02d}/{num_epochs:02d}, loss={epoch_loss / len(graphs):.4f}")
    return gnn


# --------------------------------------------------------------------------- #
# 4.  Convenience wrapper for a quick experiment
# --------------------------------------------------------------------------- #
def run_experiment() -> None:
    """
    Run a toy experiment that trains the GNN on a handful of synthetic graphs
    and prints the learned parameters for the first graph.
    """
    graphs = generate_synthetic_graphs(
        num_graphs=5,
        num_nodes=8,
        node_features=4,
        num_qubits=qlm.NUM_QUBITS,
    )
    gnn = train_gnn_and_qml(graphs, num_epochs=15, lr=0.005)
    # Inspect the parameters for the first graph
    first_graph = graphs[0]
    adj = torch.tensor(first_graph["adj"], dtype=torch.float32)
    features = torch.tensor(first_graph["features"], dtype=torch.float32)
    pred_params = gnn(features, adj)
    print("Predicted variational parameters:", pred_params.detach().cpu().numpy())


if __name__ == "__main__":
    run_experiment()
