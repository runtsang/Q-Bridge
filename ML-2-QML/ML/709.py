import torch
from torch import nn

class QCNNModel(nn.Module):
    """Extended classical QCNN with residual connections, dropout, and deeper layers."""

    def __init__(self) -> None:
        super().__init__()
        # Feature extraction
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        # Convolutional layers with residuals
        self.conv1 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.pool1 = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv4 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.pool2 = nn.Sequential(nn.Linear(16, 8), nn.ReLU())
        # Output head
        self.head = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.dropout1(x)
        conv2_out = self.conv2(x)
        # Residual connection
        conv2_out = conv2_out + x
        x = self.pool1(conv2_out)
        x = self.conv3(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.pool2(x)
        out = self.head(x)
        return torch.sigmoid(out)

    def freeze(self) -> None:
        """Freeze all parameters to prevent training."""
        for param in self.parameters():
            param.requires_grad = False

    def to_device(self, device: torch.device) -> None:
        """Move the model to the specified device."""
        self.to(device)

def QCNN() -> QCNNModel:
    """Factory returning the configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
