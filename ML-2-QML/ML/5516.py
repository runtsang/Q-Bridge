import torch
import torch.nn as nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """Hybrid convolutional network that fuses classical convolutional blocks
    with a fully‑connected backbone inspired by QCNN and Quantum‑NAT designs.
    The head can be swapped between a simple linear layer or a quantum
    expectation module when the QML variant is used.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 1 for grayscale images).
    num_classes : int
        Number of output classes (default 2 for binary classification).
    use_quantum_head : bool
        If ``True`` the class will expect a ``quantum_head`` callable
        (provided by the QML module) to be passed during ``forward``.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 use_quantum_head: bool = False) -> None:
        super().__init__()
        self.use_quantum_head = use_quantum_head

        # Convolutional feature extractor (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Fully‑connected projection (QCNN style)
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),      # 4‑dimensional feature vector
        )

        # Classical head for binary classification
        self.classifier = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, quantum_head=None) -> torch.Tensor:
        # Feature extraction
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.use_quantum_head:
            if quantum_head is None:
                raise ValueError("quantum_head callable must be provided when "
                                 "use_quantum_head=True")
            # Pass the 4‑dim features to the quantum circuit
            x = quantum_head(x)
        else:
            x = self.classifier(x)

        # Binary classification output
        return torch.cat([x, 1 - x], dim=-1)

__all__ = ["QCNNHybrid"]
