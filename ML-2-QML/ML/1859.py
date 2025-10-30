"""Hybrid autoencoder with optional quantum decoder.

Provides:
- HybridAutoencoderNet: classical encoder + quantum decoder or classical decoder.
- train_hybrid_autoencoder: training loop using torch optimizers.
- QuantumDecoder: nn.Module that wraps a PennyLane QNode.

"""

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple
import pennylane as qml

# Configuration dataclass
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quantum_decoder: bool = True
    qnn_reps: int = 2  # repetitions for the variational ansatz

# Classical decoder fallback
class ClassicalDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# Quantum decoder implementation
class QuantumDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, reps: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.reps = reps
        # Build PennyLane qnode
        self.num_wires = max(latent_dim, output_dim)
        self.dev = qml.device("default.qubit", wires=self.num_wires)
        self.qnode = self._build_qnode()
        # Parameters for the ansatz
        num_params = reps * self.num_wires * 3
        self.params = nn.Parameter(torch.randn(num_params))

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def decoder_qnode(latents: torch.Tensor, params: torch.Tensor):
            # Encode latent values into first latent_dim wires
            for i in range(self.latent_dim):
                qml.RY(latents[i], wires=i)
            # Variational ansatz
            qml.templates.StronglyEntanglingLayers(params, wires=range(self.num_wires))
            # Output: expectation of PauliZ on first output_dim wires
            return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]
        return decoder_qnode

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.qnode(latents, self.params)

# Main hybrid autoencoder
class HybridAutoencoderNet(nn.Module):
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_encoder(config)
        self.decoder = ClassicalDecoder(config.latent_dim, config.input_dim) if not config.use_quantum_decoder \
            else QuantumDecoder(config.latent_dim, config.input_dim, config.qnn_reps)

    def _build_encoder(self, config: HybridAutoencoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# Factory function mirroring the original API
def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_quantum_decoder: bool = True,
    qnn_reps: int = 2,
) -> HybridAutoencoderNet:
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quantum_decoder=use_quantum_decoder,
        qnn_reps=qnn_reps,
    )
    return HybridAutoencoderNet(config)

# Training loop
def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in dataset:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset.dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "HybridAutoencoderNet", "train_hybrid_autoencoder", "HybridAutoencoderConfig"]
