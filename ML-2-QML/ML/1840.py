import torch
import torch.nn as nn
from typing import List

class QuantumClassifierModel(nn.Module):
    """
    Classical feed‑forward classifier with optional quantum‑inspired feature branch.

    The architecture consists of:
        - Encoder: maps raw features to an intermediate representation.
        - Optional quantum‑inspired branch that applies sinusoidal embeddings
          and a lightweight network to generate additional features.
        - Classifier head: combines both branches and produces class logits.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        use_quantum_branch: bool = False,
        quantum_feature_dim: int = 8,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.use_quantum_branch = use_quantum_branch
        self.device = device

        # Primary encoder
        encoder_layers: List[nn.Module] = [
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum‑inspired feature branch
        if use_quantum_branch:
            # Sinusoidal embeddings: sin and cos of scaled inputs
            self.quantum_embedding = nn.Sequential(
                nn.Linear(num_features, quantum_feature_dim),
                nn.Sigmoid(),
                nn.Linear(quantum_feature_dim, quantum_feature_dim),
                nn.Sigmoid(),
            )
            # Small network to process the quantum embeddings
            self.quantum_branch = nn.Sequential(
                nn.Linear(quantum_feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
        else:
            self.quantum_branch = nn.Identity()

        # Classifier head
        classifier_input_dim = 128 + (32 if use_quantum_branch else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Logits of shape (batch_size, 2).
        """
        encoded = self.encoder(x)
        if self.use_quantum_branch:
            quantum_emb = self.quantum_embedding(x)
            quantum_feats = self.quantum_branch(quantum_emb)
            combined = torch.cat([encoded, quantum_feats], dim=1)
        else:
            combined = encoded
        logits = self.classifier(combined)
        return logits

__all__ = ["QuantumClassifierModel"]
