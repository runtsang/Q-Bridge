import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------
# Classical quanvolution filter (seeded from reference pair 1)
# -----------------------------------------------
class QuanvolutionFilter(nn.Module):
    """Purely classical 2×2 stride‑2 convolution followed by flattening."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.shape[0], -1)


# -----------------------------------------------
# Classical sampler network (seeded from reference pair 2)
# -----------------------------------------------
class SamplerModule(nn.Module):
    """Soft‑max sampler that produces auxiliary weights from the feature vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * 14 * 14, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


# -----------------------------------------------
# Quantum components (imported from the QML module)
# -----------------------------------------------
# The QML module will expose: QuanvolutionFilterQuantum, QuantumKernel, QCNNQuantumQNN
from.Quanvolution__gen040_qml import (
    QuanvolutionFilterQuantum,
    QuantumKernel,
    QCNNQuantumQNN,
)

# -----------------------------------------------
# Hybrid model that stitches the classical and quantum parts together
# -----------------------------------------------
class QuanvolutionHybridModel(nn.Module):
    """
    Hybrid neural network that:
      1. Extracts classical and quantum features via two parallel quanvolution filters.
      2. Generates auxiliary weights using a sampler network.
      3. Embeds the combined features into a quantum‑kernel space.
      4. Classifies the final representation with a QCNN quantum classifier.
    """
    def __init__(
        self,
        num_classes: int = 10,
        prototype_count: int = 8,
    ) -> None:
        super().__init__()

        # Feature extraction
        self.classical_filter = QuanvolutionFilter()
        self.quantum_filter = QuanvolutionFilterQuantum()

        # Sampler for adaptive weighting
        self.sampler = SamplerModule()

        # Quantum kernel for similarity embedding
        self.kernel = QuantumKernel()

        # QCNN quantum classifier
        self.qnn_classifier = QCNNQuantumQNN()

        # Prototype vectors for kernel similarity (fixed random)
        self.register_buffer(
            "prototypes",
            torch.randn(prototype_count, 4 * 14 * 14),
        )

        # Linear projection to reduce the feature dimension to the QCNN input size (8)
        # The combined feature size is 4*14*14 (classical) + 4*14*14 (quantum)
        # + 2 (sampler) + 8 (kernel similarities) = 1578
        self.pre_classifier = nn.Linear(1578, 8)

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical and quantum features
        cls_feat = self.classical_filter(x)          # (B, 784)
        q_feat = self.quantum_filter(x)             # (B, 784)

        # Concatenate raw features
        raw_features = torch.cat([cls_feat, q_feat], dim=1)   # (B, 1568)

        # Auxiliary weights from sampler
        aux_weights = self.sampler(raw_features)               # (B, 2)

        # Kernel similarities to prototypes
        sims = torch.stack(
            [self.kernel(raw_features, proto.unsqueeze(0)) for proto in self.prototypes],
            dim=1,
        )  # (B, prototype_count)

        # Final feature vector
        combined = torch.cat([raw_features, aux_weights, sims], dim=1)  # (B, 1578)

        # Reduce to QCNN input size
        reduced = self.pre_classifier(combined)                 # (B, 8)

        # Quantum QCNN classifier
        logits = self.qnn_classifier(reduced)                   # (B, num_classes)

        return logits


__all__ = ["QuanvolutionFilter", "SamplerModule", "QuanvolutionHybridModel"]
