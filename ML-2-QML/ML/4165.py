from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple
import networkx as nx
import numpy as np

# ----------------------------------------------------------------------
# Autoencoder utilities (adapted from reference 2)
# ----------------------------------------------------------------------
class AutoencoderConfig:
    """Configuration for the fully‑connected auto‑encoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Simple MLP auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int,...] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    """Convenience factory mirroring the original seed."""
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# ----------------------------------------------------------------------
# Simple graph convolutional layer (classical GNN)
# ----------------------------------------------------------------------
class GCNLayer(nn.Module):
    """One layer of a graph‑convolutional network."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj is assumed to be row‑normalised
        support = self.linear(x)
        out = torch.matmul(adj, support)
        return torch.relu(out)

# ----------------------------------------------------------------------
# Fraud detection pipeline
# ----------------------------------------------------------------------
class FraudDetectionModel(nn.Module):
    """
    Classical fraud‑detection model that combines an auto‑encoder
    for feature compression with a lightweight graph neural network.
    """
    def __init__(self,
                 autoencoder_cfg: AutoencoderConfig,
                 gnn_hidden_dims: Tuple[int,...],
                 num_classes: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(autoencoder_cfg.input_dim,
                                       latent_dim=autoencoder_cfg.latent_dim,
                                       hidden_dims=autoencoder_cfg.hidden_dims,
                                       dropout=autoencoder_cfg.dropout)

        layers = []
        in_dim = autoencoder_cfg.latent_dim
        for h in gnn_hidden_dims:
            layers.append(GCNLayer(in_dim, h, dropout))
            in_dim = h
        self.gcn_layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor
            Raw transaction features (N, F).
        adjacency : torch.Tensor
            Normalised adjacency matrix of the transaction graph (N, N).
        """
        latent = self.autoencoder.encode(features)
        h = latent
        for layer in self.gcn_layers:
            h = layer(h, adjacency)
        logits = self.classifier(h)
        return logits

    def fidelity_adjacency(self,
                           states: torch.Tensor,
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> torch.Tensor:
        """
        Construct an adjacency matrix from pairwise cosine similarities
        of latent states, a classical analogue of the quantum fidelity
        based graph used in reference 3.
        """
        n = states.size(0)
        adj = torch.zeros((n, n), dtype=torch.float32, device=states.device)
        norms = torch.norm(states, dim=1, keepdim=True) + 1e-12
        cos = torch.mm(states, states.t()) / (norms @ norms.t())
        mask1 = cos >= threshold
        adj[mask1] = 1.0
        if secondary is not None:
            mask2 = (cos >= secondary) & (~mask1)
            adj[mask2] = secondary_weight
        return adj

def train_fraud_detection(model: FraudDetectionModel,
                          data: torch.Tensor,
                          adjacency: torch.Tensor,
                          *,
                          epochs: int = 100,
                          batch_size: int = 64,
                          lr: float = 1e-3,
                          weight_decay: float = 0.0,
                          device: torch.device | None = None) -> list[float]:
    """
    Training loop for the hybrid classical model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data, adjacency)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_features, batch_adj in loader:
            batch_features = batch_features.to(device)
            batch_adj = batch_adj.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features, batch_adj)
            # Dummy labels for illustration; replace with real labels in practice
            labels = torch.zeros(batch_features.size(0), dtype=torch.long, device=device)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_features.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "GCNLayer",
    "FraudDetectionModel",
    "train_fraud_detection",
]
