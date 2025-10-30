"""Hybrid transformer integrating classical transformer, LSTM, Quanvolution, and graph operations.

The class exposes a unified API that can be configured to use optional
Quanvolution for image inputs, a classical LSTM for sequence modeling,
and a simple graph‑convolution on the hidden representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Basic building blocks -------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ----- Optional modules -------------------------------------------

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution followed by flattening."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QLSTM(nn.Module):
    """Classical LSTM cell implemented with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# ----- Graph utilities -------------------------------------------

def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-12)
    b_norm = b / (torch.norm(b, dim=-1, keepdim=True) + 1e-12)
    return torch.matmul(a_norm, b_norm.transpose(-2, -1))

def _graph_convolution(states: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Simple weighted sum of neighbor states."""
    return torch.matmul(adjacency, states)

# ----- Hybrid Transformer -------------------------------------------

class HybridTransformer(nn.Module):
    """Unified transformer that can process text or images,
    optionally augment with a classical LSTM and a graph convolution
    over the hidden representations.
    """
    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
        use_quanvolution: bool = False,
        use_lstm: bool = False,
        use_graph: bool = False,
        image_input: bool = False,
    ):
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.use_lstm = use_lstm
        self.use_graph = use_graph
        self.image_input = image_input

        if image_input:
            self.feature_extractor = QuanvolutionFilter()
            feature_dim = 4 * 14 * 14
            self.token_embedding = nn.Linear(feature_dim, embed_dim)
        else:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for token inputs")
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)

        if use_lstm:
            self.lstm = QLSTM(embed_dim, embed_dim)
        else:
            self.lstm = None

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.image_input:
            features = self.feature_extractor(x)
            tokens = self.token_embedding(features.unsqueeze(1))
            seq_len = 1
        else:
            tokens = self.token_embedding(x)
            seq_len = x.size(1)

        x = self.pos_encoder(tokens)
        for block in self.transformer_blocks:
            x = block(x, mask)

        if self.lstm is not None:
            lstm_out, _ = self.lstm(x)
            x = lstm_out

        if self.use_graph:
            B, T, E = x.shape
            states = x.reshape(B * T, E)
            adj = _cosine_similarity(states, states)
            adj.fill_diagonal_(0)
            adj_sum = adj.sum(dim=-1, keepdim=True) + 1e-12
            adj = adj / adj_sum
            states = _graph_convolution(states, adj)
            x = states.reshape(B, T, E)

        x = self.dropout(x.mean(dim=1))
        logits = self.classifier(x)
        return logits
