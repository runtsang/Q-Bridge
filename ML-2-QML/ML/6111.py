import torch
from torch import nn
import numpy as np
import hybrid_qml

class QCNNModel(nn.Module):
    """Classical convolution-inspired feature extractor mirroring QCNN."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridEstimatorQNN(nn.Module):
    """Hybrid classical‑quantum regressor combining QCNN feature extraction with a variational QNN."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = QCNNModel()
        self.quantum_estimator = hybrid_qml.HybridEstimatorQNN()
        self.readout = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through classical layers, quantum layer, and final readout."""
        # Classical feature extraction
        features = self.feature_extractor(inputs)  # shape (batch, 8)
        # Quantum evaluation
        quantum_outputs = []
        for i in range(features.shape[0]):
            # Map each feature to the corresponding input parameter
            param_dict = {p: float(features[i, j]) for j, p in enumerate(self.quantum_estimator.input_params)}
            out = self.quantum_estimator(param_dict)  # numpy array of shape (1,)
            quantum_outputs.append(out)
        quantum_tensor = torch.tensor(np.array(quantum_outputs), dtype=torch.float32)
        # Readout
        return self.readout(quantum_tensor)
