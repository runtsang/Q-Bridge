import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Optional

from.GraphQNN import state_fidelity, fidelity_adjacency
from.EstimatorQNN import EstimatorQNN

class SamplerQNN__gen158(nn.Module):
    """
    Hybrid sampler network that combines:
    * A transformer encoder for sequence modelling.
    * A feed‑forward classifier producing probability distributions.
    * An embedded regressor for auxiliary predictions.
    * Graph‑based regularisation using state fidelity.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        num_blocks: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Transformer encoder layers
        self.transformer = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    activation="relu",
                )
                for _ in range(num_blocks)
            ]
        )
        # Classifier head
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        # Embedded regressor
        self.regressor = EstimatorQNN()
        # Store hidden activations for graph construction
        self._last_hidden: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer and classifier.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, embed_dim)
            Input feature sequence.

        Returns
        -------
        probs : Tensor of shape (batch, num_classes)
            Softmax probability distribution.
        """
        # Transformer encoding
        hidden = self.transformer(x)  # (batch, seq_len, embed_dim)
        pooled = hidden.mean(dim=1)  # (batch, embed_dim)
        self._last_hidden = pooled.detach().clone()

        # Classification
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        return probs

    def regressor_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the same input through the embedded regressor.
        The regressor expects a 2‑D tensor; we use the first token.
        """
        return self.regressor(x[:, 0, :])

    def adjacency_graph(
        self,
        threshold: float = 0.8,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted adjacency graph from the last hidden states using
        fidelity.  Must call ``forward`` before invoking this method.

        Returns
        -------
        graph : networkx.Graph
            Weighted graph where edge weights are 1.0 for fidelities above
            ``threshold`` and ``secondary_weight`` for fidelities in the
            secondary range.
        """
        if self._last_hidden is None:
            raise RuntimeError("No hidden states available. Run forward first.")
        states = [h for h in self._last_hidden]
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = ["SamplerQNN__gen158"]
