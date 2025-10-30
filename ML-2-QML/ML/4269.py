import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple

class HybridFCL(nn.Module):
    """
    Classical hybrid layer that emulates a fully‑connected quantum layer,
    a sampler network, and a simple classifier.

    The forward pass returns three outputs:
        1. The encoded expectation value (shape (batch,1)).
        2. Softmax probabilities from the sampler head.
        3. Logits from the classifier head.

    The ``run`` method keeps backward compatibility with the anchor FCL
    by accepting an iterable of thetas and returning the encoded expectation
    as a NumPy array.
    """

    def __init__(self,
                 n_features: int = 1,
                 sampler_hidden: int = 4,
                 classifier_depth: int = 1) -> None:
        super().__init__()
        # Core fully‑connected layer
        self.encoder = nn.Linear(n_features, 1)

        # Sampler head: two‑dimensional softmax inspired by SamplerQNN
        self.sampler = nn.Sequential(
            nn.Linear(2, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2)
        )

        # Simple classifier head
        layers = [nn.Linear(1, n_features), nn.ReLU()]
        for _ in range(classifier_depth - 1):
            layers.extend([nn.Linear(n_features, n_features), nn.ReLU()])
        layers.append(nn.Linear(n_features, 2))
        self.classifier = nn.Sequential(*layers)

    def forward(self, thetas: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Convert input parameters to tensor
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        encoded = torch.tanh(self.encoder(x))

        # Prepare sampler input: duplicate encoded value to match 2‑dimensional head
        sampler_input = torch.cat([encoded, encoded], dim=1)
        sampler_probs = F.softmax(self.sampler(sampler_input), dim=-1)

        # Classifier logits
        logits = self.classifier(encoded)

        return (encoded.detach().numpy(),
                sampler_probs.detach().numpy(),
                logits.detach().numpy())

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Back‑compatible run method returning the encoded expectation."""
        encoded, _, _ = self.forward(thetas)
        return encoded.squeeze()

__all__ = ["HybridFCL"]
