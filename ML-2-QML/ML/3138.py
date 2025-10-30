"""Hybrid QCNN: classical graph backbone + quantum convolutional layers.

The module exposes a single class `QCNNGraphHybrid` that contains:
  * A PyTorch graph neural network that learns a weighted adjacency from the
    training data.
  * A Qiskit variational ansatz that mirrors the classical graph convolution
    pattern.
  * A `forward` method that first propagates samples through the graph GNN,
    then feeds the resulting embeddings into the quantum circuit via a
    feature‑map, and finally returns the quantum measurement expectation.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, ParameterVector
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Classical GNN
class _GraphGNN(nn.Module):
    """Simple GNN that learns node embeddings and a weighted adjacency."""
    def __init__(self, num_nodes: int, hidden_dim: int = 16):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_emb = nn.Linear(num_nodes, hidden_dim)
        self.node_emb = nn.Linear(hidden_dim, hidden_dim)
        self.weight_adj = nn.Parameter(torch.rand(num_nodes, num_nodes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, nx.Graph]:
        edge_repr = self.edge_emb(x)
        node_repr = self.node_emb(edge_repr)
        adj = self.weight_adj.clamp(min=0.0)
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        for i, j in itertools.combinations(range(self.num_nodes), 2):
            if adj[i, j] > 0.1:
                graph.add_edge(i, j, weight=adj[i, j].item())
        return node_repr, graph

# Quantum convolution / pooling primitives
def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def _conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        sub = _conv_circuit(params[i:i+3])
        qc.append(sub, [i, i+1])
    return qc

def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for i, (s, t) in enumerate(zip(sources, sinks)):
        sub = _pool_circuit(params[i*3:i*3+3])
        qc.append(sub, [s, t])
    return qc

class QCNNGraphHybrid(nn.Module):
    """Hybrid quantum‑classical convolution network."""
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int | None = None,
        num_qubits: int = 8,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = num_qubits
        self.gnn = _GraphGNN(num_nodes, hidden_dim)
        self.num_qubits = num_qubits
        # Feature map parameters
        self.feature_map = ParameterVector("φ", length=num_qubits)
        # Build ansatz
        self.circuit = QuantumCircuit(num_qubits)
        # First conv + pool
        self.circuit.compose(_conv_layer(num_qubits, "c1"), list(range(num_qubits)), inplace=True)
        self.circuit.compose(_pool_layer(list(range(0, num_qubits, 2)),
                                        list(range(1, num_qubits, 2)), "p1"),
                            list(range(num_qubits)), inplace=True)
        # Second conv + pool on half qubits
        half = num_qubits // 2
        self.circuit.compose(_conv_layer(half, "c2"), list(range(half)), inplace=True)
        self.circuit.compose(_pool_layer(list(range(half)), [half], "p2"),
                            list(range(half)), inplace=True)
        # Final measurement
        self.measure = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
        # Estimator
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.measure,
            input_params=self.feature_map,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical GNN
        node_emb, _ = self.gnn(x)
        # Flatten embeddings to match num_qubits
        flat = node_emb.view(-1, self.num_qubits)
        # Prepare input parameters for QNN
        inputs = flat.detach().cpu().numpy()
        # Run quantum circuit
        result = self.qnn.predict(inputs)
        return torch.tensor(result, dtype=torch.float32).unsqueeze(-1)

__all__ = ["QCNNGraphHybrid"]
