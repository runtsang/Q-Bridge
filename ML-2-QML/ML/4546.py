import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
import numpy as np

class SamplerQNN(nn.Module):
    """
    Classical hybrid sampler that mimics the quantum SamplerQNN.
    It combines:
    - a small MLP to produce sampler parameters,
    - a linear layer to approximate quantum sampling,
    - an optional regression head,
    - optional LSTM for sequence processing,
    - graph utilities to compute fidelity‑based adjacency.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 4,
                 n_qubits: int = 2,
                 use_lstm: bool = False,
                 use_graph: bool = False,
                 regression_head: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_lstm = use_lstm
        self.use_graph = use_graph
        self.regression_head = regression_head

        # MLP to produce sampler parameters
        self.param_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),  # 4 parameters for 2‑qubit circuit
        )

        # Linear layer to approximate sampling distribution
        self.sample_net = nn.Linear(4, 2 ** n_qubits)

        if regression_head:
            self.reg_head = nn.Linear(2 ** n_qubits, 1)

        if use_lstm:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape (batch, input_dim) or (batch, seq_len, input_dim)
        Returns:
            dict with keys:
                'probs'      – sampling distribution probabilities,
               'regression' – optional regression output,
                'graph'      – optional adjacency graph.
        """
        if self.use_lstm:
            x, _ = self.lstm(x)

        params = self.param_net(x)
        probs = F.softmax(self.sample_net(params), dim=-1)

        out = {"probs": probs}

        if self.regression_head:
            reg = self.reg_head(probs)
            out["regression"] = reg.squeeze(-1)

        if self.use_graph:
            probs_np = probs.detach().cpu().numpy()
            graph = nx.Graph()
            graph.add_nodes_from(range(probs_np.shape[0]))
            for i, j in itertools.combinations(range(probs_np.shape[0]), 2):
                fid = np.dot(probs_np[i], probs_np[j]) ** 2
                if fid > 0.5:
                    graph.add_edge(i, j, weight=1.0)
            out["graph"] = graph

        return out

    def generate_data(self, num_samples: int = 1000):
        """
        Generate synthetic data for regression: states and labels.
        """
        x = np.random.uniform(-1.0, 1.0, size=(num_samples, self.input_dim)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return torch.tensor(x), torch.tensor(y)

__all__ = ["SamplerQNN"]
