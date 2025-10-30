import torch
import torch.nn as nn
import torch.nn.functional as F

def build_classical_classifier(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Construct a feed‑forward classifier that mirrors a depth‑2 quantum ansatz and returns
    the network, the encoding indices, the weight statistics and a dummy observable list.
    The implementation adds dropout for regularisation, a subtle change from the original
    helper that keeps the public API intact while improving generalisation."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_stats = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(), nn.Dropout(p=0.1)])
        weight_stats.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_stats.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_stats, observables

class QFCModel(nn.Module):
    """Classical CNN + MLP hybrid inspired by the Quantum‑NAT architecture.

    The model first extracts spatial features with a shallow convolutional backbone,
    then projects them into a 4‑dimensional embedding.  This embedding is fed into a
    quantum‑inspired MLP (built with `build_classical_classifier`) that outputs
    classification logits.  The architecture preserves the original 4‑feature
    interface while adding a 2‑class head for downstream tasks.
    """

    def __init__(self, num_features: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_features),
        )
        self.norm = nn.BatchNorm1d(num_features)
        self.classifier, _, _, _ = build_classical_classifier(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 4‑dimensional embedding."""
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        out = self.norm(out)
        return out

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2‑class logits derived from the 4‑dimensional embedding."""
        embedding = self.forward(x)
        logits = self.classifier(embedding)
        return logits

__all__ = ["QFCModel", "build_classical_classifier"]
