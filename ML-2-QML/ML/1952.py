import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """
    A purely classical neural network that mimics the structure of the original
    QCNN but with several modern enhancements:
        * Residual skip connections between convolution layers.
        * Batch‑normalisation after each linear block.
        * Dropout for regularisation.
        * Flexible input dimension (default 8 → 16 features).
    The forward pass returns a single‑output probability via sigmoid.
    """

    def __init__(self, input_dim: int = 8, hidden_dims: tuple[int,...] = (16, 16, 12, 8, 4, 4)) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Feature mapping
        self.feature_map = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Tanh(),
        )

        # Convolution‑pooling blocks with residuals
        self.blocks = nn.ModuleList()
        in_dim = hidden_dims[0]
        for i, out_dim in enumerate(hidden_dims[1:]):
            block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Tanh(),
                nn.Dropout(p=0.2),
            )
            self.blocks.append(block)
            in_dim = out_dim

        # Final head
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor of shape (batch, input_dim)
            Input feature vector.

        Returns
        -------
        torch.Tensor of shape (batch, 1)
            Probability output in (0, 1).
        """
        x = self.feature_map(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            # Residual addition only if dimensions match
            if residual.shape == x.shape:
                x = x + residual
        return torch.sigmoid(self.head(x))

def QCNNHybridFactory() -> QCNNHybrid:
    """
    Factory returning a default configuration of :class:`QCNNHybrid`.
    """
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
