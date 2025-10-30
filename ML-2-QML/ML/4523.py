"""Classical fraud detection model that combines LSTM, graph embeddings, and a hybrid activation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEmbedder(nn.Module):
    """Simple graph neural network using adjacency matrix and linear layers."""
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # node_features: (N, in_features)
        # adjacency: (N, N)
        h = self.linear1(node_features)
        h = torch.matmul(adjacency, h)
        h = F.relu(h)
        h = self.linear2(h)
        return h


class FraudDetectionHybrid(nn.Module):
    """Hybrid classical fraud detection model with LSTM and graph embeddings."""
    def __init__(self,
                 input_dim: int,
                 lstm_hidden: int,
                 graph_in: int,
                 graph_hidden: int,
                 graph_out: int,
                 fc_hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.graph_embedder = GraphEmbedder(graph_in, graph_hidden, graph_out)
        self.fc1 = nn.Linear(lstm_hidden + graph_out, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                seq: torch.Tensor,
                graph_features: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        seq : torch.Tensor
            Transaction sequence of shape (batch, seq_len, input_dim).
        graph_features : torch.Tensor
            Node features of graph shape (N, graph_in).
        adjacency : torch.Tensor
            Adjacency matrix shape (N, N).
        Returns
        -------
        torch.Tensor
            Fraud probability of shape (batch,).
        """
        batch_size = seq.size(0)
        lstm_out, _ = self.lstm(seq)
        lstm_last = lstm_out[:, -1, :]
        graph_emb = self.graph_embedder(graph_features, adjacency)
        graph_emb = graph_emb.mean(dim=0, keepdim=True).expand(batch_size, -1)
        combined = torch.cat([lstm_last, graph_emb], dim=1)
        h = F.relu(self.fc1(combined))
        logits = self.fc2(h)
        probs = self.sigmoid(logits).squeeze(-1)
        return probs


__all__ = ["FraudDetectionHybrid"]
