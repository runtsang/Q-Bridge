import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

# ------------------------------------------------------------------
# Classical LSTM cell (drop‑in replacement for nn.LSTM)
# ------------------------------------------------------------------
class QLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), \
               torch.zeros(batch_size, self.hidden_dim, device=device)

# ------------------------------------------------------------------
# Classical self‑attention helper
# ------------------------------------------------------------------
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) /
                               np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)

# ------------------------------------------------------------------
# Classical quanvolution filter
# ------------------------------------------------------------------
class QuanvolutionFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

# ------------------------------------------------------------------
# Classical sampler network
# ------------------------------------------------------------------
class SamplerQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

# ------------------------------------------------------------------
# Hybrid LSTM‑tagger
# ------------------------------------------------------------------
class HybridQLSTM(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_quanvolution: bool = False,
                 use_self_attention: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional quantum‑inspired feature extractor
        self.use_quanvolution = use_quanvolution
        self.feature_extractor = QuanvolutionFilter() if use_quanvolution else nn.Identity()

        # LSTM core – classical by default, quantum gate logic can be swapped in
        self.lstm = QLSTM(embedding_dim, hidden_dim) if n_qubits == 0 else QLSTM(embedding_dim, hidden_dim)

        # Optional self‑attention
        self.use_self_attention = use_self_attention
        self.attention = ClassicalSelfAttention(embedding_dim) if use_self_attention else None

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.sampler = SamplerQNN()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)

        if self.use_quanvolution:
            embeds = self.feature_extractor(embeds)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        if self.use_self_attention:
            lstm_out = self.attention(lstm_out)

        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

__all__ = ["HybridQLSTM", "QLSTM", "ClassicalSelfAttention",
           "QuanvolutionFilter", "SamplerQNN"]
