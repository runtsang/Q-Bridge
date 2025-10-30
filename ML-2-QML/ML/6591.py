import torch
from torch import nn

class ResidualConnection(nn.Module):
    """Adds a linear skip connection from input to output of a block."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.skip = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x)

class QCNNGenModel(nn.Module):
    """Hybrid QCNN‑inspired classical network with residual, batch‑norm and dropout support.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dim : int
        Width of intermediate linear layers.
    num_layers : int
        Number of convolution‑pooling blocks.
    dropout : float
        Dropout probability applied after each block.
    use_batchnorm : bool
        Whether to insert BatchNorm1d after activations.
    residual : bool
        Whether to add skip connections between consecutive blocks.
    output_dim : int
        Size of the output vector (1 for binary classification).
    """
    def __init__(self,
                 input_dim: int = 8,
                 hidden_dim: int = 16,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 use_batchnorm: bool = True,
                 residual: bool = True,
                 output_dim: int = 1) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.residual = residual

        layers = []

        # Feature map
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())

        in_dim = hidden_dim
        for i in range(num_layers):
            # Convolution block
            conv = nn.Linear(in_dim, hidden_dim)
            block = [conv]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(hidden_dim))
            block.append(nn.ReLU())
            if residual:
                block.append(ResidualConnection(in_dim, hidden_dim))
            layers.extend(block)
            layers.append(nn.Dropout(dropout))

            # Pooling block
            pool = nn.Linear(hidden_dim, hidden_dim // 2)
            block = [pool]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(hidden_dim // 2))
            block.append(nn.ReLU())
            layers.extend(block)
            layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim // 2

        # Final head
        layers.append(nn.Linear(in_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.model[-1].out_features == 1:
            return torch.sigmoid(out)
        return out

def QCNNGen() -> QCNNGenModel:
    """Factory that returns a ready‑to‑train QCNNGenModel."""
    return QCNNGenModel()

__all__ = ["QCNNGenModel", "QCNNGen", "ResidualConnection"]
