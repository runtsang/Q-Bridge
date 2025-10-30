"""UnifiedSelfAttentionQNN – classical component.

The module exposes a single `UnifiedSelfAttentionQNN` class that
* builds a classical self‑attention block (using a small MLP for the
  query/key/value projections),
* feeds the attention output into a graph‑structured neural network
  that is parameterised by a list of weight matrices,
* provides a `train_step` helper that performs a single gradient update
  with PyTorch autograd.
"""

import numpy as np
import torch
from torch import nn
import networkx as nx
from typing import Sequence, Tuple, List

Tensor = torch.Tensor


class UnifiedSelfAttentionQNN(nn.Module):
    """
    A hybrid classical‑quantum‑inspired architecture.

    Parameters
    ----------
    embed_dim : int
        Dimension of the embedding space.
    qnn_arch : Sequence[int]
        Layer sizes for the graph‑structured neural network.
    """

    def __init__(self, embed_dim: int, qnn_arch: Sequence[int]):
        super().__init__()
        self.embed_dim = embed_dim
        self.qnn_arch = list(qnn_arch)

        # ----- Classical self‑attention -------------------------------------------------
        # Linear layers for query, key and value
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # ----- Graph‑structured neural network ------------------------------------------
        # Each layer is a simple tanh‑activated linear map
        self.gnn_layers = nn.ModuleList()
        for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:]):
            self.gnn_layers.append(nn.Linear(in_f, out_f, bias=False))

    # --------------------------------------------------------------------------- #
    # Classical attention
    # --------------------------------------------------------------------------- #
    def _attention(self, x: Tensor) -> Tensor:
        """
        Compute a standard dot‑product attention over the batch.
        """
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        scores = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, V)

    # --------------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------------- #
    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input batch of shape (batch_size, embed_dim).

        Returns
        -------
        * attention output : Tensor
            The result of the attention mechanism.
        * graph state : Tensor
            The graph‑structured state after the last GNN layer.
        """
        attn_out = self._attention(x)

        # propagate through the graph‑structured network
        state = attn_out
        for layer in self.gnn_layers:
            state = torch.tanh(layer(state))

        return attn_out, state

    # --------------------------------------------------------------------------- #
    # Training helper
    # --------------------------------------------------------------------------- #
    def train_step(self,
                   data_loader,
                   optimizer,
                   loss_fn,
                   device: str = "cpu") -> Tuple[Tensor, float]:
        """
        Perform one epoch of training on the supplied data_loader.

        The routine is intentionally lightweight: it simply iterates over
        batches, computes the loss, back‑propagates and steps the optimizer.
        """
        self.train()
        total_loss = 0.0
        for batch, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            attn, state = self(x=inputs)
            # a simple regression loss on the final graph state
            loss = loss_fn(state, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return torch.tensor(total_loss / (batch + 1)), total_loss

    # --------------------------------------------------------------------------- #
    # Utility: build a fidelity‑based graph from final states
    # --------------------------------------------------------------------------- #
    def build_fidelity_graph(self,
                             states: Sequence[Tensor],
                             threshold: float,
                             *,
                             secondary: float | None = None,
                             secondary_weight: float = 0.5) -> nx.Graph:
        """
        Build an undirected weighted graph where edges are defined by
        the cosine similarity (a classical proxy for fidelity) between
        the two states.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, s_i in enumerate(states):
            for j, s_j in enumerate(states[i + 1:], i + 1):
                cos = torch.dot(s_i, s_j) / (torch.norm(s_i) * torch.norm(s_j) + 1e-12)
                fid = cos.item() ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph
