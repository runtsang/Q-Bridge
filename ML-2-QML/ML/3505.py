import numpy as np
import torch
from torch import nn


class QCNNModel(nn.Module):
    """QCNN‑style fully‑connected network used as a classical surrogate."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridFCL_QCNN(nn.Module):
    """
    Classical hybrid of a fully‑connected layer with a QCNN‑style network.
    The `run` method accepts a flat list of weights that drives the
    feature‑map, the QCNN block and the final linear output.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        # Map the raw input to an 8‑dim feature vector
        self.feature_map = nn.Sequential(nn.Linear(n_features, 8), nn.Tanh())
        # QCNN surrogate
        self.qcnn = QCNNModel()
        # Final linear read‑out
        self.output = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        x = self.feature_map(inputs)
        x = self.qcnn(x)
        x = self.output(x)
        return torch.sigmoid(x)

    def run(self, thetas: list[float]) -> np.ndarray:
        """
        Run a forward pass with parameters supplied as a flat list.
        The list is interpreted as follows (assuming default n_features=1):
            * first 8 values → biases of the feature‑map linear layer
            * next 1 value   → bias of the output layer
        Remaining weights are left as in the module's learned parameters.
        """
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)

        # Feature‑map linear layer (bias only for simplicity)
        bias_fmap = values[:8]
        x = self.feature_map.weight @ torch.ones((1, 1))
        x = x + bias_fmap
        x = torch.tanh(x)

        # QCNN forward (expects batch dimension)
        x = x.t()  # shape (1, 8)
        x = self.qcnn(x)

        # Output linear layer
        bias_out = values[-1]
        x = self.output.weight @ x + bias_out
        return torch.sigmoid(x).detach().numpy()
