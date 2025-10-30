"""Hybrid classical LSTM with optional quantum gate emulation and convolutional preprocessing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFilter(nn.Module):
    """Drop‑in 2‑D convolutional filter emulating the quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Expect data shape (batch, 1, H, W) or (H, W)
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # (batch, 1)


class QLSTMGen121(nn.Module):
    """Classical LSTM with linear gates that can be swapped for quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
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
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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


class LSTMTaggerGen121(nn.Module):
    """Sequence tagging model that uses convolutional preprocessing and a hybrid LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        if n_qubits > 0:
            self.lstm = QLSTMGen121(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)
        batch_size, seq_len, dim = embeds.size()
        sqrt_dim = int(dim**0.5)
        if sqrt_dim * sqrt_dim!= dim:
            raise ValueError("embedding_dim must be a perfect square for the conv filter")
        conv_input = embeds.view(batch_size * seq_len, 1, sqrt_dim, sqrt_dim)
        conv_out = self.conv(conv_input).view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(conv_out.view(seq_len, batch_size, -1))
        tag_logits = self.hidden2tag(lstm_out.view(seq_len, -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTMGen121", "LSTMTaggerGen121"]
