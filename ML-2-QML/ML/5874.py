import torch
from torch import nn

class EnhancedQCNN(nn.Module):
    """
    A hybrid classical‑quantum inspired network that extends the original QCNN.
    The model contains an embedding, a stack of classical blocks that mimic
    the parameter count of the quantum convolution layers, and a final
    linear head.  All operations are differentiable and can be trained
    with any PyTorch optimiser.
    """

    def __init__(self,
                 embed_dim: int = 16,
                 num_layers: int = 3,
                 hidden_dim: int = 32,
                 seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.embedding = nn.Sequential(
            nn.Linear(8, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )

        self.conv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, embed_dim),
                nn.Tanh()
            )
            self.conv_blocks.append(block)

        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.conv_blocks:
            x = block(x)
        logits = self.head(x)
        return torch.sigmoid(logits)

def QCNN() -> EnhancedQCNN:
    """
    Factory that returns a default configured EnhancedQCNN.
    """
    return EnhancedQCNN()
