"""AutoencoderGen215 – Classical encoder with variational latent and interface to quantum decoder.

The encoder maps input x to latent parameters μ and logσ.
The latent z is sampled via re‑parameterisation.
Training uses reconstruction loss computed by a quantum decoder function that
takes z and returns a probability vector over the input dimension.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class AutoencoderGen215Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.encoder = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(in_dim, latent_dim)
        self.logvar_layer = nn.Linear(in_dim, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between the latent posterior and the unit Gaussian."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def train_autoencoder(
    model: AutoencoderGen215Net,
    quantum_decoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Trains the VAE where quantum_decoder(z) returns a probability vector
    of the same shape as the input. Reconstruction loss is cross‑entropy
    between the input (treated as one‑hot) and the decoder output.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reconstruction_loss_fn = nn.CrossEntropyLoss(reduction="sum")

    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z, mu, logvar = model(batch)
            probs = quantum_decoder(z)  # shape (batch, input_dim)
            targets = batch.argmax(dim=1)
            recon_loss = reconstruction_loss_fn(probs, targets) / batch.size(0)
            loss = recon_loss + kl_loss(mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(loader))
    return history

__all__ = ["AutoencoderGen215Net", "kl_loss", "train_autoencoder"]
