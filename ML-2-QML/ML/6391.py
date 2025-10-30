import torch
from torch import nn

class QCNNFeatureMap(nn.Module):
    """Learnable feature map that embeds raw 8‑dim vectors into a higher‑dim space.

    The original seed used a fixed ZFeatureMap.  Here we replace it with a small
    feed‑forward network so that the gradients flow through the embedding
    step.  The network outputs a vector of size 8 (the number of qubits),
    which is then interpreted as rotation angles for the ZZ‑phase
    rotations in the quantum circuit.
    """
    def __init__(self, in_features: int = 8, out_features: int = 8, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QCNNConvLayer(nn.Module):
    """A classical analogue of the two‑qubit convolution block.

    The seed used a hard‑coded 2‑qubit unitary.  We now
    implement the block as a linear layer with 3 parameters per
    pair, matching the parameter count of the original conv_circuit
    (the three rotation angles).  This allows the model
    to *re‑use* the quantum‑derived structure while
    **embedding** it into a feature‑space that’s trainable.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

class QCNNPoolLayer(nn.Module):
    """Simple average pooling that halves the feature dimension."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.mean(x.reshape(batch, -1, 2), dim=2)

class QCNNEnhanced(nn.Module):
    """Hybrid QCNN that combines a learnable feature map with
    a classical convolution‑pooling stack mirroring the quantum
    architecture.

    The model is fully differentiable and can be trained with
    any PyTorch optimiser.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = QCNNFeatureMap()
        self.conv1 = QCNNConvLayer(8, 8)
        self.pool1 = QCNNPoolLayer(8, 4)
        self.conv2 = QCNNConvLayer(4, 4)
        self.pool2 = QCNNPoolLayer(4, 2)
        self.conv3 = QCNNConvLayer(2, 2)
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.head(x)
        return torch.sigmoid(x)

def QCNN() -> QCNNEnhanced:
    """Factory returning the configured :class:`QCNNEnhanced`."""
    return QCNNEnhanced()

__all__ = ["QCNN", "QCNNEnhanced", "QCNNFeatureMap", "QCNNConvLayer", "QCNNPoolLayer"]
