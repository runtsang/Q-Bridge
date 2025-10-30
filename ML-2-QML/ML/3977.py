from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

class HybridQLSTM(nn.Module):
    """Classical LSTM with optional quantum‑style scaling of gate activations."""
    def __init__(self, input_dim: int, hidden_dim: int, use_quantum: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_quantum = use_quantum

        # Standard linear gates
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if use_quantum:
            # Parameters mimicking quantum non‑linearities
            self.q_bias = nn.Parameter(torch.randn(hidden_dim))
            self.q_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            if self.use_quantum:
                # Quantum‑inspired scaling
                f = torch.tanh(f * self.q_scale + self.q_bias)
                i = torch.tanh(i * self.q_scale + self.q_bias)
                g = torch.tanh(g * self.q_scale + self.q_bias)
                o = torch.tanh(o * self.q_scale + self.q_bias)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Sequence tagging model that employs HybridQLSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, use_quantum: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, use_quantum=use_quantum)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

def build_classifier_circuit(num_features: int, depth: int
                            ) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
    """Construct a classical feed‑forward classifier mirroring the quantum variant."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables
