"""Hybrid classical LSTM that optionally uses quantum‑style gates,
convolution, and self‑attention.

The public API matches the original QLSTM module, so existing code
can import :class:`QLSTM` and :class:`LSTMTagger` unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
#  Classical LSTM cell (drop‑in replacement for the quantum gates)
# ------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Classical LSTM cell with the same interface as the quantum version.
    The cell can be used directly or wrapped inside a :class:`LSTMTagger`.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections for the four gates
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            states: Optional (hx, cx) tuple
        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            (hx, cx): Final hidden and cell states
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# ------------------------------------------------------------------
#  Conv filter (classical emulation of a quantum quanvolution)
# ------------------------------------------------------------------
class ConvFilter(nn.Module):
    """
    Simple 2×2 convolution that mimics the behaviour of the quantum
    quanvolution used in the seed.  It produces a scalar activation
    per input patch which can be concatenated with the embeddings.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# ------------------------------------------------------------------
#  Classical self‑attention helper
# ------------------------------------------------------------------
class ClassicalSelfAttention:
    """
    Lightweight self‑attention that mirrors the interface of the
    quantum circuit.  It is used as a drop‑in pre‑processing step.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# ------------------------------------------------------------------
#  Tagger that chains embeddings → conv → attention → LSTM
# ------------------------------------------------------------------
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that accepts an optional quantum‑style
    preprocessing pipeline before feeding data into the LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Pre‑processing blocks
        self.conv = ConvFilter()
        self.attn = ClassicalSelfAttention(embed_dim=4)

        # Choose between classical and quantum LSTM
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: Tensor of word indices, shape (seq_len, batch)
        Returns:
            log‑probabilities over tags for each token
        """
        # Word embeddings
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed_dim)

        # Conv filter applied to each embedding (treated as 2×2 patch)
        conv_feats = torch.tensor(
            [self.conv.run(e.cpu().numpy().reshape(2, 2)) for e in embeds],
            dtype=torch.float32,
        ).unsqueeze(-1)  # (seq_len, batch, 1)

        # Self‑attention over the conv output
        rotation = np.zeros(12)          # 4 qubits × 3 rotation params
        entangle = np.zeros(3)           # 3 entanglement params
        attn_out = self.attn.run(rotation, entangle, conv_feats.numpy())
        attn_tensor = torch.from_numpy(attn_out).float().unsqueeze(-1)  # (seq_len, batch, 1)

        # Concatenate all features
        lstm_input = torch.cat([embeds, conv_feats, attn_tensor], dim=-1)

        # LSTM (classical or quantum)
        lstm_out, _ = self.lstm(lstm_input.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
