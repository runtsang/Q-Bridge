import torch
import torch.nn as nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QFCModel(nn.Module):
    """CNN followed by a fully connected projection to four features."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class HybridQuantumClassifier(nn.Module):
    """
    Classical binary classifier that fuses QCNN and QFC feature extractors
    with a final linear head. The model is fully differentiable and
    suitable for end‑to‑end training.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor from QuantumNAT
        self.cnn = QFCModel().features
        # Linear mapping from CNN output to the 8‑dimensional input of QCNN
        self.lin_to_qcnn = nn.Linear(16 * 7 * 7, 8)
        # QCNN classifier
        self.qcnn = QCNNModel()
        # Final linear layer to produce a logit
        self.classifier = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, H, W)
        features = self.cnn(x)
        flattened = features.view(features.size(0), -1)
        qcnn_input = self.lin_to_qcnn(flattened)
        qcnn_output = self.qcnn(qcnn_input)
        # qcnn_output already in [0,1] due to sigmoid
        probs = qcnn_output
        return torch.cat((probs, 1 - probs), dim=-1)
