import torch
from torch import nn

# Import the classical transformer block defined in the repository
from QTransformerTorch import TransformerBlockClassical

class QCNNHybrid(nn.Module):
    """Hybrid classical QCNN model with optional transformer augmentation."""
    def __init__(
        self,
        use_transformer: bool = False,
        transformer_heads: int = 4,
        transformer_ffn: int = 64,
        transformer_blocks: int = 2,
    ) -> None:
        super().__init__()
        # Classical feature extraction
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Optional transformer block
        if use_transformer:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockClassical(
                        embed_dim=4,
                        num_heads=transformer_heads,
                        ffn_dim=transformer_ffn,
                    )
                    for _ in range(transformer_blocks)
                ]
            )
        else:
            self.transformer = None
        # Final classification head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical QCNN pipeline."""
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        if self.transformer is not None:
            # Reshape for transformer: (batch, seq_len, embed_dim)
            x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)
        return torch.sigmoid(self.head(x))

__all__ = ["QCNNHybrid"]
