"""Hybrid classifier that fuses classical dense layers, an auto‑encoder, and an optional quantum feature extractor.

The module is a drop‑in replacement for the original `QuantumClassifierModel.py`.  It is fully classical and can
optionally delegate the feature extraction to a quantum routine supplied by the companion QML module.  The design
borrows the depth‑controlled network from the original classifier, the auto‑encoder architecture from the
`Autoencoder.py` seed, and the fidelity‑graph utilities from the graph QNN seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import networkx as nx


# --------------------------------------------------------------------------- #
# Utility functions – copied and adapted from the seed modules
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def _random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def _fidelity_graph(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# --------------------------------------------------------------------------- #
# Auto‑encoder building block – from reference 2
# --------------------------------------------------------------------------- #
@dataclass
class _AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class _AutoencoderNet(nn.Module):
    """Lightweight fully‑connected auto‑encoder used for pre‑training."""

    def __init__(self, cfg: _AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# Depth‑controlled classifier network – from reference 1
# --------------------------------------------------------------------------- #
def _build_classifier_network(num_features: int, depth: int) -> nn.Module:
    layers: List[nn.Module] = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))  # binary head for compatibility
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Unified classifier
# --------------------------------------------------------------------------- #
class UnifiedClassifierQNN(nn.Module):
    """
    A hybrid classifier that can operate purely classically or delegate feature extraction to a quantum routine.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input.
    num_classes : int, default 2
        Number of target classes.
    latent_dim : int, default 32
        Size of the latent vector produced by the auto‑encoder.
    hidden_dims : tuple[int, int], default (128, 64)
        Hidden layer sizes for the auto‑encoder.
    depth : int, default 3
        Depth of the classical feed‑forward classifier.
    use_qnn : bool, default False
        When ``True`` the model expects a quantum feature extractor.
    quantum_feature_extractor : Callable[[torch.Tensor], torch.Tensor] | None, default None
        Function that converts a batch of raw inputs into a torch tensor of quantum‑derived features.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        depth: int = 3,
        *,
        use_qnn: bool = False,
        quantum_feature_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.use_qnn = use_qnn
        self.quantum_feature_extractor = quantum_feature_extractor

        # Auto‑encoder for pre‑training / feature extraction
        cfg = _AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        self.autoencoder = _AutoencoderNet(cfg)

        # Classical feed‑forward classifier
        self.classifier = _build_classifier_network(latent_dim, depth)

        # Final linear head – output dimensionality depends on the chosen path
        out_features = 2 if use_qnn else latent_dim
        self.final_linear = nn.Linear(out_features, num_classes)

    # --------------------------------------------------------------------- #
    # Helper – pre‑train the auto‑encoder
    # --------------------------------------------------------------------- #
    @staticmethod
    def pretrain_autoencoder(
        model: _AutoencoderNet,
        data: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> List[float]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            optimizer.zero_grad()
            recon = model(data)
            loss = loss_fn(recon, data)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        return history

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_qnn:
            if self.quantum_feature_extractor is None:
                raise ValueError("Quantum feature extractor function must be supplied when use_qnn=True.")
            features = self.quantum_feature_extractor(x)
        else:
            latent = self.autoencoder.encode(x)
            features = self.classifier(latent)
        logits = self.final_linear(features)
        return logits


# --------------------------------------------------------------------------- #
# Example usage (to be removed from production code)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import numpy as np

    # Dummy data
    X = torch.randn(32, 20)
    y = torch.randint(0, 2, (32,))

    # Instantiate model
    model = UnifiedClassifierQNN(input_dim=20, num_classes=2, use_qnn=False)

    # Pre‑train auto‑encoder
    history = UnifiedClassifierQNN.pretrain_autoencoder(model.autoencoder, X, epochs=10)

    # Forward pass
    logits = model(X)
    print("Logits shape:", logits.shape)
