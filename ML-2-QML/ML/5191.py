"""HybridTransformer: a modular transformer that can operate on token sequences,
regression data, graph embeddings, or fraud features using a purely classical
implementation."""

from __future__ import annotations

import math
import itertools
from typing import Optional, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
#  Data utilities – regression, graph, fraud
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with a sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

def random_graph_network(num_nodes: int, samples: int) -> tuple[Sequence[int], list[list[torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random graph network for testing."""
    arch = [num_nodes] + [num_nodes] * 2
    weights = [torch.randn(arch[i], arch[i+1]) for i in range(len(arch)-1)]
    target_weight = weights[-1]
    training_data = [(torch.randn(arch[0]), target_weight @ torch.randn(arch[0])) for _ in range(samples)]
    return arch, weights, training_data, target_weight

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Construct a weighted adjacency graph based on state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
#  Core transformer components
# --------------------------------------------------------------------------- #

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]

class TransformerBlockClassical(nn.Module):
    """Standard multi‑head attention + feed‑forward block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class GraphAttentionBlock(nn.Module):
    """Simplified graph‑based attention using adjacency."""
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out   = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # adjacency: (batch, N, N)
        q = self.query(x).view(x.size(0), -1, self.num_heads, x.size(-1) // self.num_heads).transpose(1,2)
        k = self.key(x).view(x.size(0), -1, self.num_heads, x.size(-1) // self.num_heads).transpose(1,2)
        v = self.value(x).view(x.size(0), -1, self.num_heads, x.size(-1) // self.num_heads).transpose(1,2)
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        attn_scores = attn_scores.masked_fill(adjacency.unsqueeze(1) == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1,2).contiguous().view(x.size(0), -1, x.size(-1))
        return self.out(out)

class FraudEncoder(nn.Module):
    """Classical fraud‑detection encoder mirroring the photonic circuit."""
    def __init__(self, params: FraudLayerParameters):
        super().__init__()
        self.model = build_fraud_detection_program(params, [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --------------------------------------------------------------------------- #
#  Hybrid transformer
# --------------------------------------------------------------------------- #

class HybridTransformer(nn.Module):
    """Transformer that can process text, regression, graph, or fraud data."""
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_graph: bool = False,
        use_fraud: bool = False,
        fraud_params: Optional[FraudLayerParameters] = None,
    ) -> None:
        super().__init__()
        self.use_graph = use_graph
        self.use_fraud = use_fraud
        self.num_classes = num_classes
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.token_embedding = nn.Linear(1, embed_dim)  # fallback for regression
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        if use_graph:
            self.graph_encoder = nn.Linear(embed_dim, embed_dim)
        else:
            self.graph_encoder = None
        if use_fraud:
            assert fraud_params is not None, "Fraud parameters required when use_fraud=True"
            self.fraud_encoder = FraudEncoder(fraud_params)
        else:
            self.fraud_encoder = None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None,
                fraud_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding
        if isinstance(self.token_embedding, nn.Embedding):
            tokens = self.token_embedding(x)
        else:
            tokens = self.token_embedding(x.unsqueeze(-1))
        x = self.pos_embedding(tokens)
        # Transformer blocks
        x = self.transformers(x)
        # Optional graph bias
        if self.graph_encoder is not None and adjacency is not None:
            graph_emb = self.graph_encoder(adjacency.mean(dim=1).unsqueeze(-1))
            x = x + graph_emb
        # Optional fraud bias
        if self.fraud_encoder is not None and fraud_features is not None:
            fraud_emb = self.fraud_encoder(fraud_features)
            x = x + fraud_emb.unsqueeze(1)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)
