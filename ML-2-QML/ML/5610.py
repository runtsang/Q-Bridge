import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class HybridClassifier(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 input_size: tuple[int, int] = (28, 28),
                 cnn_channels: int = 8,
                 transformer_heads: int = 4,
                 transformer_layers: int = 2,
                 transformer_ffn_dim: int = 128,
                 random_unitary_dim: int = 64):
        super().__init__()
        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Compute flattened dimension after CNN
        dummy = torch.zeros(1, input_channels, *input_size)
        with torch.no_grad():
            dummy_feat = self.features(dummy)
        flattened_dim = dummy_feat.numel()
        # Random unitary-inspired feature transform (fixed)
        self.random_unitary = nn.Linear(flattened_dim, random_unitary_dim, bias=False)
        nn.init.orthogonal_(self.random_unitary.weight)
        self.random_unitary.weight.requires_grad = False
        # Transformer encoder
        self.pos_encoder = PositionalEncoder(random_unitary_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=random_unitary_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        # Classification head
        self.classifier = nn.Linear(random_unitary_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        batch = features.size(0)
        flattened = features.view(batch, -1)
        transformed = self.random_unitary(flattened)
        # Add positional encoding
        pos = self.pos_encoder(transformed.unsqueeze(1))  # (batch, seq_len=1, dim)
        transformer_out = self.transformer(pos)
        out = transformer_out.mean(dim=1)
        logits = self.classifier(out)
        return logits
