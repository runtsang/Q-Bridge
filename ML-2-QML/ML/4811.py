import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

__all__ = ["FCLGen265"]

class FCLGen265(nn.Module):
    """
    Hybrid classical model that combines:
      * A convolutional backbone inspired by Quantum‑NAT
      * A fully‑connected “quantum‑like” layer that uses a tanh‑based
        expectation value computed from a user supplied parameter vector
      * A lightweight sampler head that turns the 4‑dimensional output
        into a probability distribution over two classes.
    The design mirrors the structure of the original FCL example while
    adding a richer feature extractor and probabilistic output.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Flattened fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)
        # Sampler head
        self.sampler = nn.Sequential(
            nn.Linear(2, 4), nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return batch‑wise 4‑dimensional features."""
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        out = self.fc(flat)
        return self.norm(out)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute a “quantum expectation” from a list of parameters.
        The method is a lightweight stand‑in for a parameterised quantum
        circuit that would otherwise be evaluated on a backend.
        """
        theta = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.fc[2](theta)).mean(dim=0)
        return expectation.detach().cpu().numpy()

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the sampler head to the 4‑dimensional features.
        Returns a probability distribution over two classes.
        """
        feat = self.forward(x)
        logits = self.sampler(feat)
        return F.softmax(logits, dim=-1)
