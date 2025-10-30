"""
Hybrid LSTM tagger – classical implementation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical self‑attention head
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """
    A simple dot‑product self‑attention block.
    Projects inputs into query, key, and value spaces and returns
    the attended representation.
    """
    def __init__(self, embed_dim: int, attn_dim: int | None = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim or embed_dim
        self.q_proj = nn.Linear(embed_dim, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.attn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, embed_dim)
        Returns:
            attended: (seq_len, batch, attn_dim)
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.attn_dim), dim=-1)
        return torch.matmul(scores, v)


# --------------------------------------------------------------------------- #
#  Classical sampler module (soft‑max based)
# --------------------------------------------------------------------------- #
class SamplerModule(nn.Module):
    """
    Small feed‑forward network that produces a probability distribution
    over the input vector.  Used to emulate a quantum sampler in the
    classical baseline.
    """
    def __init__(self, in_features: int = 2, hidden: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


# --------------------------------------------------------------------------- #
#  Classical estimator (regression head)
# --------------------------------------------------------------------------- #
class EstimatorModule(nn.Module):
    """
    Small regression network that maps a vector to a scalar.
    """
    def __init__(self, in_features: int = 2, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Classical Q‑LSTM (drop‑in replacement)
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """
    Classic LSTM cell whose gates are realised by linear layers.
    Mirrors the original quantum interface but remains fully classical.
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None) \
            -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(self.dropout(hx.unsqueeze(0)))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None) \
            -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))


# --------------------------------------------------------------------------- #
#  Hybrid tagger that stitches together all heads
# --------------------------------------------------------------------------- #
class HybridLSTMTagger(nn.Module):
    """
    Sequence‑tagging model that combines:
      * Classical / quantum self‑attention
      * Sampler network (probabilistic re‑weighting)
      * Estimator network (optional regression head)
      * LSTM (classical or quantum)
      * Final classification layer
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = ClassicalSelfAttention(embedding_dim) if not use_quantum else QuantumSelfAttention(n_qubits)
        self.sampler = SamplerModule()
        self.estimator = EstimatorModule()
        self.lstm = QLSTM(embedding_dim, hidden_dim) if not use_quantum else QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns:
            logits: (seq_len, tagset_size)
            regression: (batch, 1)
        """
        embeds = self.embedding(sentence)          # (seq_len, batch, embed_dim)
        # Self‑attention
        attn_out = self.attention(embeds)          # (seq_len, batch, attn_dim)
        # Combine context and original embeddings
        combined = torch.cat([embeds, attn_out], dim=-1)
        # Sampler re‑weights the combined features
        weights = self.sampler(combined[-1])       # use last time‑step as a proxy
        weighted = combined * weights.unsqueeze(0)
        # LSTM
        lstm_out, _ = self.lstm(weighted)
        # Classification logits
        logits = F.log_softmax(self.hidden2tag(lstm_out), dim=-1)
        # Optional regression head on the final hidden state
        regression = self.estimator(lstm_out[-1])
        return {"logits": logits, "regression": regression}


__all__ = ["HybridLSTMTagger", "QLSTM", "SamplerModule", "EstimatorModule"]
