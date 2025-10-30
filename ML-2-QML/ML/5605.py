import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

def random_network(qnn_arch, samples):
    """Generate random weight matrices for a simple feedforward network."""
    weights = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(torch.randn(out_f, in_f))
    return qnn_arch, weights

def fidelity_adjacency(states, threshold, *, secondary=None, secondary_weight=0.5):
    """Build a weighted graph from pairwise cosine similarity of state vectors."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j, b in enumerate(states[i+1:], i+1):
            fid = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

class ParameterizedLayer(nn.Module):
    """Linear layer followed by tanh, scaling, and shift."""
    def __init__(self, in_features, out_features, shift=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.shift = shift
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = out * self.scale + self.bias + self.shift
        return out

class HybridBinaryClassifier(nn.Module):
    """CNN backbone followed by a graph‑aware classical head."""
    def __init__(self, adjacency_threshold=0.9, shift=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.param_layer = ParameterizedLayer(1, 1, shift=shift)
        self.adjacency_threshold = adjacency_threshold

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        features = self.fc2(x)
        graph = fidelity_adjacency(features, self.adjacency_threshold)
        x = self.fc3(features)
        x = self.param_layer(x)
        probs = torch.sigmoid(x).squeeze(-1)
        agg = probs.clone()
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                agg[node] += probs[neighbors].mean()
            deg = graph.degree[node]
            agg[node] = agg[node] / (1 + deg)
        return torch.stack([agg, 1 - agg], dim=-1)

__all__ = ["HybridBinaryClassifier"]
