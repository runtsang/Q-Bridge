import torch
import torch.nn as nn

class FraudLayer(nn.Module):
    """Linear + Tanh + trainable scale & shift."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.ones(out_features))
        self.shift = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift

class HybridNATModel(nn.Module):
    """CNN feature extractor + FraudLayer + final classifier."""
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
        # Fraud detection style layer
        self.fraud_layer = FraudLayer(in_features=16 * 7 * 7, out_features=2)
        self.final_fc = nn.Linear(2, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        fraud_out = self.fraud_layer(flattened)
        out = self.final_fc(fraud_out)
        return self.norm(out)

__all__ = ["HybridNATModel"]
