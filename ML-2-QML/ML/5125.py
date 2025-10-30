import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ----------------- Classical transformer primitives -----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer_blocks(x)
        x = self.dropout(x.mean(dim=1))
        return self.out(x)

# ----------------- Hybrid head -----------------
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

def build_classifier_network(num_features: int, depth: int) -> nn.Sequential:
    layers = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(num_features, 1))
    return nn.Sequential(*layers)

# ----------------- Main hybrid classifier -----------------
class HybridBinaryClassifier(nn.Module):
    """
    A unified binary classifier that can operate purely classically or with a
    quantum expectation head.  The feature extractor is configurable: a CNN
    back‑bone, a transformer stack, or a hybrid of both.
    """
    def __init__(self,
                 use_cnn: bool = True,
                 use_transformer: bool = False,
                 transformer_cfg: Optional[dict] = None,
                 num_features: int = 120,
                 depth: int = 3,
                 shift: float = 0.0) -> None:
        super().__init__()
        self.use_cnn = use_cnn
        self.use_transformer = use_transformer
        if use_cnn:
            # Simple CNN backbone – same topology as the seed but with
            # slightly tweaked dropout rates for stability.
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(p=0.5),
                nn.Flatten()
            )
            self.fc = nn.Sequential(
                nn.Linear(55815, num_features),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(num_features, 84),
                nn.ReLU(),
                nn.Linear(84, 1)
            )
        if use_transformer:
            if transformer_cfg is None:
                transformer_cfg = dict(
                    embed_dim=128,
                    num_heads=4,
                    num_blocks=2,
                    ffn_dim=256,
                    dropout=0.1
                )
            self.transformer = TextClassifier(**transformer_cfg)
        self.shift = shift
        self.head = HybridFunction.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cnn:
            features = self.cnn(x)
            logits = self.fc(features)
        elif self.use_transformer:
            logits = self.transformer(x)
        else:
            raise RuntimeError("At least one feature extractor must be enabled")
        probs = self.head(logits, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "PositionalEncoder",
    "TextClassifier",
    "HybridFunction",
    "build_classifier_network",
    "HybridBinaryClassifier"
]
