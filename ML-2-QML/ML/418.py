"""Hybrid classical LSTM module with optional quantum mixing and CRF decoder.

This module keeps the original API (`QLSTM` and `LSTMTagger`) but adds a
mixing hyper‑parameter `alpha` that blends a purely classical linear
projection of each gate with a quantum‑style embedding (here simulated by
an additional linear layer).  The tagging head optionally attaches a
conditional random field (CRF) decoder to capture label dependencies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from torchcrf import CRF  # optional dependency
except ImportError:  # pragma: no cover
    CRF = None  # type: ignore[assignment]

class QLSTM(nn.Module):
    """
    Hybrid LSTM cell that mixes quantum‑style gate outputs with classical
    linear projections.

    Parameters
    ----------
    input_dim : int
        Size of the input vector.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Number of qubits used for the quantum component.  The value is
        only used for signature compatibility; the quantum part is
        simulated by an additional linear layer.
    alpha : float, default=1.0
        Mixing ratio. 0.0 → pure quantum, 1.0 → pure classical.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.alpha = float(alpha)

        # classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # quantum‑style embedding (simulated)
        self.quantum_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # classical gates
            f_cls = torch.sigmoid(self.forget_linear(combined))
            i_cls = torch.sigmoid(self.input_linear(combined))
            g_cls = torch.tanh(self.update_linear(combined))
            o_cls = torch.sigmoid(self.output_linear(combined))

            # quantum‑style gates (simulated)
            q = self.quantum_linear(combined)
            f_q = torch.sigmoid(q)
            i_q = torch.sigmoid(q)
            g_q = torch.tanh(q)
            o_q = torch.sigmoid(q)

            # mixing
            f = self.alpha * f_cls + (1 - self.alpha) * f_q
            i = self.alpha * i_cls + (1 - self.alpha) * i_q
            g = self.alpha * g_cls + (1 - self.alpha) * g_q
            o = self.alpha * o_cls + (1 - self.alpha) * o_q

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can optionally attach a CRF decoder.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Size of LSTM hidden states.
    vocab_size : int
        Vocabulary size.
    tagset_size : int
        Number of distinct tags.
    n_qubits : int, default=0
        Number of qubits for the hybrid LSTM.  0 selects a pure
        `nn.LSTM`.
    alpha : float, default=1.0
        Mixing ratio for the hybrid LSTM.
    use_crf : bool, default=False
        If True and `torchcrf` is available, a CRF layer is appended.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        alpha: float = 1.0,
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, alpha)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.use_crf = use_crf and CRF is not None
        if self.use_crf:
            self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape (seq_len,) or (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag scores or CRF decoded tag indices.
        """
        embeds = self.word_embeddings(sentence)
        # ensure shape (seq_len, batch, embed_dim)
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(1)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        if self.use_crf:
            # CRF expects scores of shape (batch, seq_len, tagset)
            scores = tag_logits.permute(1, 0, 2)  # (batch, seq, tag)
            decoded = self.crf.decode(scores)
            return torch.tensor(decoded, device=sentence.device, dtype=torch.long)
        return F.log_softmax(tag_logits, dim=2)

__all__ = ["QLSTM", "LSTMTagger"]
