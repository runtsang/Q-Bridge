"""Hybrid LSTM with quantum-inspired gating for sequence tagging."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RandomFourierFeatureLayer(nn.Module):
    """Random Fourier feature mapping to emulate quantum kernels."""
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * np.sqrt(2 * gamma))
        self.b = nn.Parameter(torch.rand(output_dim) * 2 * np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        z = torch.matmul(x, self.W) + self.b
        return torch.cat([torch.cos(z), torch.sin(z)], dim=1)

class HybridQLSTM(nn.Module):
    """Classical LSTM with quantum-inspired random Fourier feature gating."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, feature_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim or 4 * input_dim

        # Random Fourier feature layers for input and hidden state
        self.input_feat = RandomFourierFeatureLayer(input_dim, self.feature_dim)
        self.hidden_feat = RandomFourierFeatureLayer(hidden_dim, self.feature_dim)

        # Linear layers mapping concatenated features to gate logits
        self.forget_linear = nn.Linear(2 * self.feature_dim, hidden_dim)
        self.input_linear = nn.Linear(2 * self.feature_dim, hidden_dim)
        self.update_linear = nn.Linear(2 * self.feature_dim, hidden_dim)
        self.output_linear = nn.Linear(2 * self.feature_dim, hidden_dim)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            # Encode input and hidden state into higher-dimensional feature space
            x_feat = self.input_feat(x)
            hx_feat = self.hidden_feat(hx)
            combined = torch.cat([x_feat, hx_feat], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

class HybridLSTMTagger(nn.Module):
    """Sequence tagging model using HybridQLSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
