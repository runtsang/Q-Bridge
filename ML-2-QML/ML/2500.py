from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical transformer components (from reference 2)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding size does not match layer embedding size")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        d_k = embed_dim // self.num_heads
        k = k.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        attn_output = torch.matmul(scores, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.combine_heads(attn_output)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, num_blocks: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.pos_encoder = PositionalEncoder(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        return x

# --------------------------------------------------------------------------- #
# Hybrid head components
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class DenseHybrid(nn.Module):
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class QCNet(nn.Module):
    """
    Hybrid CNN with optional transformer encoder and a hybrid head.
    The head can be a dense sigmoid (classical) or a quantum circuit (see qml_code).
    """
    def __init__(
        self,
        use_transformer: bool = False,
        transformer_params: dict | None = None,
        head_type: str = "dense",
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        # Convolutional backbone (identical to seed 1)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Optional transformer encoder
        if use_transformer:
            if transformer_params is None:
                transformer_params = {
                    "embed_dim": 15,
                    "num_heads": 3,
                    "ffn_dim": 120,
                    "num_blocks": 2,
                    "dropout": 0.1,
                }
            self.transformer = TransformerEncoder(**transformer_params)
        else:
            self.transformer = None

        # Hybrid head
        if head_type == "dense":
            self.hybrid = DenseHybrid(1, shift=shift)
        else:
            raise NotImplementedError("Only dense head is available in the classical module")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        # Apply transformer if enabled
        if self.transformer:
            seq_len = x.size(1) // 15
            x = x.view(x.size(0), seq_len, 15)
            x = self.transformer(x)
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)

# Alias for compatibility with the original seed
HybridQuantumBinaryClassifier = QCNet
__all__ = ["QCNet", "HybridQuantumBinaryClassifier"]
