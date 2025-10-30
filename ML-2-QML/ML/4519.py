import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

class HybridFunction(nn.Module):
    """Classical approximation of a quantum expectation head."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)

class Hybrid(nn.Module):
    """Dense head that replaces a quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.activation = HybridFunction(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

class HybridQuantumBinaryClassifier(nn.Module):
    """CNN based binary classifier with a classical hybrid head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

def rbf_kernel_matrix(a: torch.Tensor, b: torch.Tensor, gamma: float = 1.0) -> np.ndarray:
    """Compute the RBF kernel matrix between two tensors."""
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    sq_norm = (diff ** 2).sum(-1)
    return np.exp(-gamma * sq_norm.numpy())

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, a in enumerate(states):
        for j in range(i + 1, len(states)):
            fid = state_fidelity(a, states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph
