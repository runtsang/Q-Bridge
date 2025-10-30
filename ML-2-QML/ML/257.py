import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d

class QFCModelEnhanced(nn.Module):
    """Hybrid classical‑quantum model with advanced features.

    The architecture consists of:
    1. Classical feature extractor (CNN) → 2. Quantum variational layer (depth‑controlled, parameter‑shared) → 3. Classical MLP head.

    This design increases expressive power while maintaining a modest parameter count.
    """

    def __init__(self, num_classes: int = 4, depth: int = 3, entanglement: str = "full") -> None:
        super().__init__()
        # 1. Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 2. Quantum variational layer placeholder (to be defined in QML module)
        self.depth = depth
        self.entanglement = entanglement
        self.n_wires = 4
        self.norm = BatchNorm1d(4)

        # 3. Classical MLP head
        self.classifier = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Extract classical features
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        # Pass through quantum layer (expects a callable from QML module)
        # In the QML implementation, this will be replaced by a quantum circuit.
        quantum_output = self.quantum_forward(flattened)
        # Classical classifier head
        out = self.classifier(quantum_output)
        return out

    def quantum_forward(self, flattened: torch.Tensor) -> torch.Tensor:
        """
        Placeholder for quantum forward pass.
        In the QML module, this will be overridden by a quantum circuit
        that returns a tensor of shape (bsz, 4).
        """
        # For compatibility, simply return the first four features.
        # The QML implementation will replace this method.
        return flattened[:, :4]
