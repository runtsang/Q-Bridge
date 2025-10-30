"""Pure PyTorch implementation of an extended LSTM with optional weight shuffling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ShuffledLinear(nn.Module):
    """Linear layer with optional weight shuffling for regularization."""
    def __init__(self, in_features: int, out_features: int, shuffle: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.shuffle = shuffle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.shuffle:
            with torch.no_grad():
                perm = torch.randperm(out.shape[-1], device=x.device)
                out = out[..., perm]
        return out


class ClassicalGate(nn.Module):
    """Wrapper for a gate implemented with a ShuffledLinear."""
    def __init__(self, in_features: int, out_features: int, shuffle: bool = False):
        super().__init__()
        self.linear = ShuffledLinear(in_features, out_features, shuffle)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QLSTM(nn.Module):
    """Classical LSTM cell with optional weight shuffling on each gate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, shuffle: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.shuffle = shuffle

        gate_dim = hidden_dim

        self.forget = ClassicalGate(input_dim + hidden_dim, gate_dim, shuffle=self.shuffle)
        self.input = ClassicalGate(input_dim + hidden_dim, gate_dim, shuffle=self.shuffle)
        self.update = ClassicalGate(input_dim + hidden_dim, gate_dim, shuffle=self.shuffle)
        self.output = ClassicalGate(input_dim + hidden_dim, gate_dim, shuffle=self.shuffle)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

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


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use the extended QLSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, shuffle=shuffle)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
