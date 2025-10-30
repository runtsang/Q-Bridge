from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFCLClassifier(nn.Module):
    """
    Classical network that emulates the quantum FCL, the layered
    classifier, and the Quantum‑NAT feature extractor.

    Attributes
    ----------
    encoding_indices : list[int]
        Indices of the qubits that the quantum circuit would encode.
    weight_sizes : list[int]
        Number of trainable parameters per layer, mirroring the quantum
        variational layers.
    observables : list[int]
        One observable per qubit (Z measurement in the quantum case).
    """

    def __init__(self, num_features: int, depth: int, conv_channels: int = 8) -> None:
        super().__init__()

        # Convolutional feature extractor (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Compute flattened feature size for a 28×28 input
        dummy = torch.zeros(1, 1, 28, 28)
        feat_size = self.features(dummy).view(1, -1).size(1)

        # Data‑encoding linear layer (mimics quantum Rx encoding)
        self.encoding = nn.Linear(feat_size, num_features)
        self.encoding_activation = nn.Tanh()

        # Deep classifier (mimics layered quantum ansatz + readout)
        layers = []
        in_dim = num_features
        weight_sizes = []
        for _ in range(depth):
            lin = nn.Linear(in_dim, num_features)
            layers.extend([lin, nn.ReLU()])
            weight_sizes.append(lin.weight.numel() + lin.bias.numel())
            in_dim = num_features
        self.classifier = nn.Sequential(*layers, nn.Linear(in_dim, 2))
        weight_sizes.append(self.classifier[-1].weight.numel() + self.classifier[-1].bias.numel())

        # Observables placeholder (one per qubit)
        observables = list(range(num_features))

        # Normalisation of logits
        self.norm = nn.BatchNorm1d(2)

        # Store metadata for interchangeability
        self.encoding_indices = list(range(num_features))
        self.weight_sizes = weight_sizes
        self.observables = observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass that mirrors the quantum encoding and classification
        pipeline.  Returns logits for binary classification.
        """
        # Convolutional feature extraction
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)

        # Encoding layer
        encoded = self.encoding_activation(self.encoding(feat))

        # Classifier
        logits = self.classifier(encoded)

        # Normalise logits
        return self.norm(logits)
