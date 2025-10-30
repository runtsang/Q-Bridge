"""Hybrid classical model combining an autoencoder and a classifier.

The module builds a lightweight PyTorch network that first compresses the
input through a fully–connected autoencoder and then classifies the latent
representation.  It mirrors the interface of the quantum helper in the
reference seeds so that the two implementations can be swapped in a
pipeline.
"""

import torch
from torch import nn
from typing import Iterable, Tuple, List, Callable

# --------------------------------------------------------------------------- #
# Autoencoder components (adapted from reference pair 2)
# --------------------------------------------------------------------------- #

class AutoencoderConfig:
    """Configuration for the autoencoder network."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder used inside the hybrid model."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def Autoencoder(input_dim: int,
                *, latent_dim: int = 32,
                hidden_dims: Tuple[int,...] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# Classifier component (adapted from reference pair 3)
# --------------------------------------------------------------------------- #

def build_classifier_network(num_features: int, depth: int) -> Tuple[nn.Module, List[int]]:
    """
    Construct a feed‑forward classifier that mirrors the quantum variant.

    Returns a sequential network and a list of feature indices used for
    the encoding (identical to the quantum encoding scheme).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)  # binary classification
    layers.append(head)
    return nn.Sequential(*layers), list(range(num_features))


# --------------------------------------------------------------------------- #
# Combined hybrid model
# --------------------------------------------------------------------------- #

class HybridModel(nn.Module):
    """
    Classic hybrid model that first auto‑encodes the input and then classifies
    the latent representation.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input.
    latent_dim : int
        Size of the latent vector produced by the encoder.
    hidden_dims : Tuple[int,...]
        Hidden layer sizes for the autoencoder.
    dropout : float
        Dropout probability used in the encoder/decoder.
    depth : int
        Number of layers in the classifier head.
    """
    def __init__(self, input_dim: int,
                 *, latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1,
                 depth: int = 2) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.classifier, _ = build_classifier_network(latent_dim, depth)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of ``x``."""
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the latent vector."""
        return self.autoencoder.decode(z)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """Logits for a binary classification task."""
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass: encode → classify.
        The reconstruction branch can be optionally used in a multi‑task loss.
        """
        z = self.encode(x)
        return self.classifier(z)


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _ensure_batch(values: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(values, torch.Tensor):
        tensor = values
    else:
        tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def train_hybrid(model: HybridModel,
                 data: torch.Tensor,
                 *,
                 epochs: int = 50,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 device: torch.device | None = None) -> List[float]:
    """
    Train the auto‑encoder and classifier jointly on ``data``.
    Dummy binary labels are used to illustrate the loss structure; in a
    real setting the labels would come from a downstream task.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_ensure_batch(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model.autoencoder(batch)
            logits = model.classifier(model.autoencoder.encode(batch))
            # Dummy binary labels (all ones) – replace with real targets as needed
            loss = loss_fn(logits, torch.ones_like(logits))
            loss += nn.MSELoss()(recon, batch)  # reconstruction loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridModel",
           "Autoencoder", "AutoencoderNet", "AutoencoderConfig",
           "build_classifier_network",
           "train_hybrid"]
