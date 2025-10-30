import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "generate_superposition_data", "fidelity_adjacency"]

def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = [0, 1]
    return network, encoding, weight_sizes, observables

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = torch.dot(s_i / (torch.norm(s_i)+1e-12), s_j / (torch.norm(s_j)+1e-12)).item() ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        node_features, adjacency = data
        hidden = F.relu(self.fc1(node_features))
        aggregated = torch.matmul(adjacency, hidden)
        out = self.fc2(aggregated)
        return out

class QuantumCircuitEncoder(nn.Module):
    def __init__(self, num_qubits: int, num_features: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_features = num_features
        self.circuit = QuantumCircuit(num_qubits)
        self.encoding_params = ParameterVector("x", num_features)
        self.weights = ParameterVector("theta", num_qubits * 2)
        for i, param in enumerate(self.encoding_params):
            self.circuit.rx(param, i)
        idx = 0
        for _ in range(2):
            for i in range(num_qubits):
                self.circuit.ry(self.weights[idx], i)
                idx += 1
            for i in range(num_qubits - 1):
                self.circuit.cz(i, i+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        expectations = []
        for i in range(batch_size):
            bound_circuit = self.circuit.bind_parameters(dict(zip(self.encoding_params, x[i])))
            state = Statevector(bound_circuit)
            exp = []
            for q in range(self.num_qubits):
                probs = state.probabilities()
                expectation = 0.0
                for idx, prob in enumerate(probs):
                    bit = (idx >> q) & 1
                    expectation += ((-1)**bit) * prob
                exp.append(expectation)
            expectations.append(exp)
        return torch.tensor(expectations, dtype=torch.float32)

class QuantumClassifierModel(nn.Module):
    def __init__(self,
                 data_type: str = "tabular",
                 num_features: int | None = None,
                 vocab_size: int | None = None,
                 num_classes: int = 2,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 ffn_dim: int = 256,
                 graph_features: int = 64,
                 n_qubits: int = 4,
                 regression: bool = False):
        super().__init__()
        self.data_type = data_type
        self.regression = regression
        self.num_classes = num_classes if not regression else 1

        if data_type == "tabular":
            layers = [nn.Linear(num_features, 64), nn.ReLU(),
                      nn.Linear(64, 32), nn.ReLU(),
                      nn.Linear(32, embed_dim)]
            self.feature_extractor = nn.Sequential(*layers)
        elif data_type == "text":
            self.pos_encoder = PositionalEncoder(embed_dim)
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim)
                  for _ in range(num_blocks)])
            self.feature_extractor = nn.Identity()
        elif data_type == "graph":
            self.gnn = GraphNeuralNetwork(input_dim=graph_features,
                                          hidden_dim=embed_dim,
                                          output_dim=embed_dim)
            self.feature_extractor = nn.Identity()
        elif data_type == "quantum":
            self.quantum_encoder = QuantumCircuitEncoder(num_qubits=n_qubits,
                                                         num_features=num_features)
            self.feature_extractor = nn.Identity()
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        if regression:
            self.head = nn.Linear(embed_dim, 1)
        else:
            self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_type == "tabular":
            h = self.feature_extractor(x)
        elif self.data_type == "text":
            h = self.pos_encoder(x)
            h = self.transformer(h)
            h = h.mean(dim=1)
        elif self.data_type == "graph":
            h = self.gnn(x)
        elif self.data_type == "quantum":
            h = self.quantum_encoder(x)
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")
        return self.head(h)
