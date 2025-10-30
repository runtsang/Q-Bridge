import torch
from torch import nn
import QCNN_QML as qml

class QCNNHybrid(nn.Module):
    """
    Hybrid classical‑quantum convolutional network that merges the
    convolution‑pooling backbone from Quantum‑NAT with a variational
    QCNN circuit from the QCNN seed. The classical branch extracts
    spatial features and projects them to a 4‑dimensional vector,
    which is fed into a 4‑qubit QCNN that outputs another 4‑dimensional
    measurement vector. The two branches are concatenated and
    passed through a final linear layer to produce the output.
    """
    def __init__(self) -> None:
        super().__init__()
        # Classical convolutional backbone (Quantum‑NAT style)
        self.classical = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Map flattened features to 4‑dimensional vector
        self.fc_to_qnn = nn.Linear(16 * 7 * 7, 4)
        # Quantum block (QCNN)
        self.qnn = qml.QCNNQuantum()
        # Final classifier
        self.final = nn.Linear(8, 1)  # 4 classical + 4 quantum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical path
        x = self.classical(x)
        x = x.view(x.size(0), -1)
        classical_features = self.fc_to_qnn(x)  # (batch, 4)

        # Quantum path
        qnn_output = self.qnn(classical_features)  # (batch, 4)

        # Concatenate and classify
        combined = torch.cat([classical_features, qnn_output], dim=1)  # (batch, 8)
        out = torch.sigmoid(self.final(combined))
        return out
