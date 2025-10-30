"""
Combined classical model that mirrors Quantum‑NAT, but with a full auto‑encoder bottleneck
and a sampler‑style classifier for probabilistic outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re‑exported helpers from the seed auto‑encoder module
try:
    from Autoencoder import AutoencoderNet
except Exception:  # pragma: no cover
    # Minimal stub if Autoencoder module is missing – keeps the module importable.
    class AutoencoderNet(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims=(128, 64)):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], input_dim)
            )

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# Main hybrid model
# --------------------------------------------------------------------------- #
class QuantumNATHybrid(nn.Module):
    """
    Classical CNN → Encoder → Quantum‑inspired layer → Auto‑encoder → Classifier.
    The “quantum” part is a thin feed‑forward block that can be swapped with a real
    quantum circuit via the QML counterpart.
    """

    def __init__(self, latent_dim: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        # 1. Convolutional feature extractor (same as seed)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 2. Encoder to latent space
        self.encoder = AutoencoderNet(input_dim=16 * 7 * 7, latent_dim=latent_dim)
        # 3. Quantum‑inspired variational block
        self.quantum_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),          # non‑linear “quantum” activation
            nn.Linear(latent_dim, latent_dim),
        )
        # 4. Decoder (reconstruction) – used only for loss if training auto‑encoder
        self.decoder = self.encoder.decoder
        # 5. Final classifier mapping latent to class logits
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning class logits.
        """
        bsz = x.shape[0]
        feat = self.cnn(x)
        flat = feat.view(bsz, -1)
        latent = self.encoder.encode(flat)
        q_out = self.quantum_layer(latent)
        logits = self.classifier(q_out)
        return self.norm(logits)

    # --------------------------------------------------------------------- #
    # Utility evaluation compatible with FastEstimator
    # --------------------------------------------------------------------- #
    def evaluate(self, parameter_sets: list[list[float]]) -> list[list[float]]:
        """
        Simple wrapper around FastEstimator that treats each parameter set
        as a batch of inputs and returns class probabilities.
        """
        from FastBaseEstimator import FastEstimator

        estimator = FastEstimator(self)
        probs = estimator.evaluate(
            observables=[lambda out: F.softmax(out, dim=-1)],
            parameter_sets=parameter_sets,
            shots=None
        )
        return probs

__all__ = ["QuantumNATHybrid"]
