import torch
import torch.nn as nn
import torch.nn.functional as F

class FCL(nn.Module):
    """
    Hybrid fully‑connected layer that combines a lightweight sampler encoder
    (inspired by SamplerQNN) with a depth‑controlled feed‑forward classifier
    (inspired by QuantumClassifierModel).  The class is fully classical and
    compatible with PyTorch workflows.
    """

    def __init__(self, num_features: int = 1, depth: int = 2, use_sampler: bool = True):
        """Build the encoder and classifier sub‑networks.

        Args:
            num_features: dimensionality of the raw input.
            depth: number of hidden layers in the classifier head.
            use_sampler: if True, prepend a 2‑to‑4‑to‑2 softmax sampler
                (mirroring the QML sampler architecture).  Otherwise the
                raw input feeds directly into the classifier.
        """
        super().__init__()
        in_dim = num_features
        if use_sampler:
            self.encoder = nn.Sequential(
                nn.Linear(num_features, 4),
                nn.Tanh(),
                nn.Linear(4, 4),
                nn.Tanh(),
            )
            in_dim = 4
        else:
            self.encoder = nn.Identity()

        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, 2))  # binary classification head
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        x = self.encoder(x)
        logits = self.classifier(x)
        return logits

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns softmax probabilities."""
        logits = self.forward(inputs)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["FCL"]
