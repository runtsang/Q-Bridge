"""Hybrid classical LSTM with optional classical convolution filter."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQLSTM(nn.Module):
    """Pure PyTorch LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            gates = torch.sigmoid(self.linear(combined))
            f, i, g, o = torch.chunk(gates, 4, dim=1)
            g = torch.tanh(g)
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

class ClassicalQuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution filter inspired by quanvolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class HybridQLSTM(nn.Module):
    """Hybrid LSTM that optionally uses a classical quanvolution filter."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_quantum_conv: bool = False,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.use_quantum_conv = use_quantum_conv
        # In the classical branch we ignore the quantum conv flag and use identity
        self.conv = ClassicalQuanvolutionFilter() if use_quantum_conv else nn.Identity()
        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        conv_features = self.conv(embeds)
        lstm_out, _ = self.lstm(conv_features.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

class LSTMTagger(nn.Module):
    """Tagger that delegates to HybridQLSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_quantum_conv: bool = False,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.model = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            vocab_size,
            tagset_size,
            use_quantum_conv,
            n_qubits,
        )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        return self.model(sentence)

__all__ = [
    "HybridQLSTM",
    "LSTMTagger",
    "ClassicalQLSTM",
    "ClassicalQuanvolutionFilter",
]
