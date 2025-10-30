from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical LSTM cell with linear gates
class ClassicalQLSTM(nn.Module):
    """Linear‑gate LSTM that mimics the quantum interface."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

# ConvFilter – classical drop‑in replacement for quanvolution
class ConvFilter(nn.Module):
    """Simple 2‑D convolution used as a lightweight pre‑processor."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

# Hybrid LSTM core
class QLSTMGen104(nn.Module):
    """
    Hybrid LSTM that can run purely classically or with quantum‑inspired gates.
    An optional ConvFilter can be attached for local feature extraction.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, conv_preprocess: bool = False) -> None:
        super().__init__()
        self.conv_preprocess = conv_preprocess
        if conv_preprocess:
            self.conv = ConvFilter()
        self.lstm = ClassicalQLSTM(input_dim, hidden_dim, n_qubits=n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_preprocess:
            # placeholder: no real preprocessing for sequence data
            pass
        return self.lstm(x)

# Sequence‑tagging wrapper
class LSTMTagger(nn.Module):
    """Wraps the hybrid LSTM for sequence tagging tasks."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, conv_preprocess: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen104(embedding_dim, hidden_dim, n_qubits=n_qubits, conv_preprocess=conv_preprocess)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(logits, dim=1)

__all__ = ["QLSTMGen104", "LSTMTagger"]
