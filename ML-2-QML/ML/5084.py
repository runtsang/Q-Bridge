import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class SelfAttentionLayer(nn.Module):
    """Classical self‑attention block."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        query = torch.matmul(inputs, rotation_params.t())
        key = torch.matmul(inputs, entangle_params.t())
        scores = F.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, inputs)

class QCNNFeatureExtractor(nn.Module):
    """QCNN‑style feature extractor built from fully‑connected layers."""
    def __init__(self, input_dim: int, depth: int = 2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.Tanh())
        self.feature_map = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_map(x)

class SamplerQNNHybrid(nn.Module):
    """Hybrid sampler‑classifier that combines classical and quantum inspired blocks."""
    def __init__(self,
                 num_features: int,
                 classifier_depth: int = 2,
                 use_qcnn: bool = True,
                 use_attention: bool = True):
        super().__init__()
        self.use_qcnn = use_qcnn
        self.use_attention = use_attention

        # Classical encoding
        self.encoder = nn.Linear(num_features, num_features)

        # Optional QCNN extractor
        if self.use_qcnn:
            self.qcnn = QCNNFeatureExtractor(num_features, depth=classifier_depth)
        else:
            self.qcnn = None

        # Classifier head
        in_dim = num_features
        if self.use_qcnn:
            in_dim = num_features  # QCNN keeps same dimensionality
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, num_features),
            nn.ReLU(),
            nn.Linear(num_features, 2)
        )

        # Optional attention
        if self.use_attention:
            self.attention = SelfAttentionLayer(embed_dim=num_features)
        else:
            self.attention = None

        # Placeholder for a quantum sampler; to be set via set_quantum_sampler
        self.quantum_sampler: Optional[callable] = None

    def set_quantum_sampler(self, sampler: callable) -> None:
        """Attach a quantum sampler callable that accepts parameters and returns a probability vector."""
        self.quantum_sampler = sampler

    def forward(self, x: torch.Tensor,
                rotation_params: Optional[np.ndarray] = None,
                entangle_params: Optional[np.ndarray] = None) -> torch.Tensor:
        """Classic forward pass producing logits."""
        x = self.encoder(x)
        if self.use_attention and rotation_params is not None and entangle_params is not None:
            # Convert to torch tensors
            rot = torch.tensor(rotation_params, dtype=torch.float32)
            ent = torch.tensor(entangle_params, dtype=torch.float32)
            x = self.attention(x, rot, ent)
        if self.use_qcnn and self.qcnn is not None:
            x = self.qcnn(x)
        logits = self.classifier(x)
        return logits

    def sample(self, params: np.ndarray) -> np.ndarray:
        """Sample from the quantum component if a sampler has been attached."""
        if self.quantum_sampler is None:
            raise RuntimeError("No quantum sampler attached.")
        return self.quantum_sampler(params)

    def evaluate(self,
                 x: torch.Tensor,
                 params: np.ndarray,
                 rotation_params: Optional[np.ndarray] = None,
                 entangle_params: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, np.ndarray]:
        """Return both classical logits and quantum sampling probabilities."""
        logits = self.forward(x, rotation_params, entangle_params)
        probs = self.sample(params)
        return logits, probs

def SamplerQNN() -> SamplerQNNHybrid:
    """Factory that returns a ready‑to‑use hybrid module."""
    return SamplerQNNHybrid(num_features=2, classifier_depth=2, use_qcnn=True, use_attention=True)

__all__ = ["SamplerQNNHybrid", "SamplerQNN"]
