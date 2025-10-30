from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from typing import Iterable

class HybridSamplerEstimatorQLSTMGraphNet(nn.Module):
    """
    Classical hybrid network that integrates a sampler, estimator, LSTM
    and graph utilities.  The architecture mirrors the quantum
    counterpart while remaining fully classical for fast experimentation.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 4,
                 n_qubits: int = 0,
                 graph_threshold: float = 0.8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical sampler: softmax over 2 outputs
        self.sampler = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

        # Classical estimator: regression network
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

        # Classical LSTM (placeholder for a quantum LSTM)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.graph_threshold = graph_threshold

    def _state_fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Squared overlap of two unit‑norm vectors."""
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def _fidelity_adjacency(self,
                            states: Iterable[torch.Tensor],
                            threshold: float,
                            *,
                            secondary: float | None = None,
                            secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def forward(self, inputs: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape (seq_len, batch, input_dim).

        Returns
        -------
        dict
            Dictionary containing sampler logits, sampler probabilities,
            estimator outputs, hidden states and a graph of state fidelities.
        """
        lstm_out, _ = self.lstm(inputs)

        sampler_logits = []
        sampler_probs = []
        estimator_out = []
        hidden_states = []

        for h in lstm_out.unbind(dim=0):
            h_slice = h[:, :self.input_dim]
            samp = self.sampler(h_slice)
            sampler_logits.append(samp)
            sampler_probs.append(F.softmax(samp, dim=-1))
            est = self.estimator(h_slice)
            estimator_out.append(est)
            hidden_states.append(h)

        sampler_logits = torch.stack(sampler_logits, dim=0)
        sampler_probs = torch.stack(sampler_probs, dim=0)
        estimator_out = torch.stack(estimator_out, dim=0)

        graph = self._fidelity_adjacency(hidden_states, self.graph_threshold)

        return {
            "sampler_logits": sampler_logits,
            "sampler_probs": sampler_probs,
            "estimator_out": estimator_out,
            "hidden_states": lstm_out,
            "fidelity_graph": graph
        }

__all__ = ["HybridSamplerEstimatorQLSTMGraphNet"]
