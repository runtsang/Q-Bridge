import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class HybridSelfAttentionClassifier:
    """
    Classical implementation of a hybrid self‑attention + classifier.
    The attention block is a standard scaled dot‑product self‑attention
    implemented with torch tensors.  The output is fed to a feed‑forward
    classifier with a user‑defined depth, mirroring the structure of
    the quantum classifier in the reference pair.
    """

    def __init__(
        self,
        num_features: int,
        attention_embed_dim: int,
        classifier_depth: int,
        classifier_num_classes: int = 2,
        device: str = "cpu",
    ):
        self.num_features = num_features
        self.attention_embed_dim = attention_embed_dim
        self.classifier_depth = classifier_depth
        self.classifier_num_classes = classifier_num_classes
        self.device = device

        # Attention parameters
        self.rotation_params = nn.Parameter(
            torch.randn(attention_embed_dim, num_features, device=device)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(attention_embed_dim, num_features, device=device)
        )

        # Classifier network
        layers = []
        in_dim = attention_embed_dim
        for _ in range(classifier_depth):
            layers.append(nn.Linear(in_dim, attention_embed_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(attention_embed_dim, classifier_num_classes))
        self.classifier = nn.Sequential(*layers).to(device)

    def attention(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Perform scaled dot‑product self‑attention on the input tensor.
        """
        query = torch.matmul(inputs, self.rotation_params.t())
        key = torch.matmul(inputs, self.entangle_params.t())
        scores = torch.softmax(
            query @ key.t() / np.sqrt(self.attention_embed_dim), dim=-1
        )
        return scores @ inputs

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: attention → classifier.
        """
        attn_out = self.attention(inputs.to(self.device))
        logits = self.classifier(attn_out)
        return logits

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray = None,
        entangle_params: np.ndarray = None,
    ) -> np.ndarray:
        """
        Convenience wrapper to run the model with NumPy arrays.
        If parameters are supplied, they override the internally
        stored parameters for a single forward pass.
        """
        if rotation_params is not None:
            self.rotation_params.data = torch.tensor(
                rotation_params.reshape(self.attention_embed_dim, self.num_features),
                dtype=torch.float32,
                device=self.device,
            )
        if entangle_params is not None:
            self.entangle_params.data = torch.tensor(
                entangle_params.reshape(self.attention_embed_dim, self.num_features),
                dtype=torch.float32,
                device=self.device,
            )
        inputs_t = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        logits = self.forward(inputs_t)
        return logits.cpu().numpy()

__all__ = ["HybridSelfAttentionClassifier"]
