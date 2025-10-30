from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#  Classical sub‑modules (extracted / adapted from the seed repo)
# --------------------------------------------------------------------------- #

def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
#  HybridModel: Classical wrapper
# --------------------------------------------------------------------------- #

class HybridModel(nn.Module):
    """
    A modular neural network that chains a fully‑connected layer,
    a self‑attention block, an autoencoder, and a classification head.
    The design mirrors the quantum construction so that the same high‑level
    flow can be replicated in a quantum implementation.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 4,
        autoencoder_cfg: dict | None = None,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        # 1. Fully‑connected layer
        self.fcl = nn.Linear(n_features, n_features)

        # 2. Self‑attention (using PyTorch's MultiheadAttention)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)

        # 3. Autoencoder
        cfg = autoencoder_cfg or {
            "input_dim": n_features,
            "latent_dim": 32,
            "hidden_dims": (128, 64),
            "dropout": 0.1,
        }
        self.autoencoder = Autoencoder(**cfg)

        # 4. Classifier
        self.classifier, _, _, _ = build_classifier_circuit(n_features, classifier_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that mimics the quantum data‑flow:
            1. Linear -> tanh
            2. Self‑attention
            3. Autoencoder
            4. Classifier
        """
        # 1. Fully‑connected + non‑linearity
        x = torch.tanh(self.fcl(x))

        # 2. Self‑attention expects (batch, seq, embed)
        seq = x.unsqueeze(1)  # add sequence dimension
        attn_output, _ = self.attn(seq, seq, seq)
        x = attn_output.squeeze(1)

        # 3. Autoencoder
        x = self.autoencoder(x)

        # 4. Classifier
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Example training loop (optional, but useful for quick experiments)
# --------------------------------------------------------------------------- #

def train_hybrid(
    model: HybridModel,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            # Dummy targets for demonstration (random)
            targets = torch.randint(0, 2, (batch.size(0),), device=device)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history
