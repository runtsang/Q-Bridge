"""HybridSamplerCNN: a classical neural network that marries a sampler sub‑network with a QCNN‑style feature extractor.

The architecture is a direct synthesis of SamplerQNN and QCNNModel.  The feature extractor emulates the 3‑layer quantum convolutional network, while the sampler sub‑network mirrors the 2‑qubit parameterised sampler from the QNN helper.  The two sub‑networks are coupled by feeding the first two convolutional features into the sampler, producing a probability distribution that is subsequently squashed to a scalar prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerCNN(nn.Module):
    """
    Classical hybrid network that combines:
    • A 3‑layer convolutional feature extractor (inspired by QCNN).
    • A 2‑qubit sampler sub‑network (inspired by SamplerQNN).
    """

    def __init__(self) -> None:
        super().__init__()

        # ----- Feature extractor -----
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # ----- Sampler sub‑network -----
        self.sampler_net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # Final classification head
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass that:
        1. Extracts features via the convolutional stack.
        2. Uses the first two feature dimensions as input to the sampler network.
        3. Returns the soft‑max probability of the first sampler output,
           optionally followed by a sigmoid for binary classification.
        """
        # Feature extraction
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Sampler input: first two feature dimensions
        sampler_input = x[:, :2]

        # Sampler output: probability distribution over two classes
        sampler_out = self.sampler_net(sampler_input)
        probs = F.softmax(sampler_out, dim=-1)

        # Optional: combine with head for a scalar prediction
        # classification = torch.sigmoid(self.head(x))
        # return classification

        return probs


__all__ = ["HybridSamplerCNN"]
