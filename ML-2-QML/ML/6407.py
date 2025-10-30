"""
Classical LSTM tagging model with optional quantum gate placeholders.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM(nn.Module):
    """
    A fully classical LSTM cell that mirrors the interface of the quantum variant.
    The gates are implemented with standard linear layers followed by sigmoid/tanh.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input vector.
        hidden_dim : int
            Dimensionality of the hidden state.
        n_qubits : int, optional
            Ignored in the classical implementation but kept for API compatibility.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)


class QLSTMTagger(nn.Module):
    """
    Sequence‑tagging model that can fall back to a classical LSTM or a quantum‑enhanced one.
    The tag head is a simple linear projection in the classical path; a quantum estimator
    can be plugged in via the :func:`EstimatorQNN` helper for research experiments.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        embedding_dim : int
            Dimension of word embeddings.
        hidden_dim : int
            Hidden size of the LSTM.
        vocab_size : int
            Size of the vocabulary.
        tagset_size : int
            Number of distinct tags.
        n_qubits : int, default 0
            If >0 the constructor keeps the signature but still uses a classical LSTM
            for maximum compatibility.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # In the pure‑classical build we ignore the qubit hint but keep the API
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of sentences.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (seq_len, batch_size) containing word indices.

        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag scores of shape (seq_len, batch_size, tagset_size).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


def EstimatorQNN() -> nn.Module:
    """
    Returns a lightweight fully‑connected regression network that mirrors the
    quantum EstimatorQNN example but stays purely classical. The network
    can be reused as a drop‑in head for regression tasks.
    """
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)

    return EstimatorNN()


__all__ = ["QLSTM", "QLSTMTagger", "EstimatorQNN"]
