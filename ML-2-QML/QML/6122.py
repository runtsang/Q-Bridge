import pennylane as qml
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoder:
    def __init__(self, cfg: HybridAutoencoderConfig):
        self.cfg = cfg
        self.dev = qml.device("default.qubit", wires=cfg.latent_dim)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.cfg.latent_dim))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.cfg.latent_dim))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.cfg.latent_dim)]
        return circuit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.shape[0], -1)
        num_layers = len(self.cfg.hidden_dims)
        weights = torch.randn((num_layers, 3, self.cfg.latent_dim))
        latent_vectors = []
        for i in range(x_flat.shape[0]):
            latent_vectors.append(self.circuit(x_flat[i], weights))
        return torch.stack(latent_vectors)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoder_layers = []
        dim = self.cfg.latent_dim
        for h in reversed(list(self.cfg.hidden_dims)):
            decoder_layers.append(nn.Linear(dim, h))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(self.cfg.dropout))
            dim = h
        decoder_layers.append(nn.Linear(dim, self.cfg.input_dim))
        decoder = nn.Sequential(*decoder_layers)
        return decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def HybridAutoencoderFactory(input_dim: int, *, latent_dim: int = 32,
                             hidden_dims: Tuple[int, int] = (128, 64),
                             dropout: float = 0.1) -> HybridAutoencoder:
    cfg = HybridAutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return HybridAutoencoder(cfg)

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory", "HybridAutoencoderConfig"]
