# Hybrid classical LSTM with optional estimator-based gate approximation.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EstimatorQNN(nn.Module):
    """Tiny feed‑forward network used to emulate a quantum gate."""
    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QLayer(nn.Module):
    """Gate layer that can be either a classical linear transform or a
    small estimator network that mimics a quantum circuit."""
    def __init__(self, n_qubits: int, use_estimator: bool) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.use_estimator = use_estimator
        if use_estimator:
            self.estimator = EstimatorQNN(input_dim=n_qubits, output_dim=n_qubits)
        else:
            self.linear = nn.Linear(n_qubits, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_estimator:
            return self.estimator(x)
        return self.linear(x)

class HybridQLSTM(nn.Module):
    """Drop‑in replacement that supports classical, quantum‑inspired
    and hybrid gate implementations."""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 use_estimator: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_estimator = use_estimator

        if n_qubits > 0 and use_estimator:
            # Quantum‑inspired gates via small estimator networks
            self.forget = QLayer(n_qubits, use_estimator=True)
            self.input = QLayer(n_qubits, use_estimator=True)
            self.update = QLayer(n_qubits, use_estimator=True)
            self.output = QLayer(n_qubits, use_estimator=True)
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            # Standard classical LSTM
            self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.n_qubits > 0 and self.use_estimator:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between hybrid and standard LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_estimator: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0 or use_estimator:
            self.lstm = HybridQLSTM(embedding_dim,
                                    hidden_dim,
                                    n_qubits=n_qubits,
                                    use_estimator=use_estimator)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
