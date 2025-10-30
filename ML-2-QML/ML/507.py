"""classical autoencoder with regularization and early stopping.

The module defines an AutoencoderNet and a factory Autoencoder that returns a configured instance.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with optional L2/ KL regularizer."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int,...] = (128, 64),
                 dropout: float = 0.1,
                 reg_type: str | None = None,
                 reg_coeff: float = 0.0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.reg_type = reg_type
        self.reg_coeff = reg_coeff

        encoder_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def _regularizer(self, z: torch.Tensor) -> torch.Tensor:
        if self.reg_type == "l2":
            return self.reg_coeff * torch.mean(z.pow(2))
        if self.reg_type == "kl":
            # KL divergence to standard normal prior
            return self.reg_coeff * torch.mean(-0.5 * (1 + torch.log(z.pow(2) + 1e-8) - z.pow(2)))
        return torch.tensor(0.0, device=z.device)

def Autoencoder(input_dim: int,
                latent_dim: int = 32,
                hidden_dims: tuple[int,...] = (128, 64),
                dropout: float = 0.1,
                reg_type: str | None = None,
                reg_coeff: float = 0.0) -> AutoencoderNet:
    """Factory that returns a configured AutoencoderNet."""
    return AutoencoderNet(input_dim, latent_dim, hidden_dims, dropout, reg_type, reg_coeff)

def train_autoencoder(model: AutoencoderNet,
                      data: torch.Tensor,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      early_stop: int | None = None,
                      device: torch.device | None = None) -> dict[str, list[float]]:
    """Train the autoencoder with optional early stopping.

    Returns a history dict with keys'recon_loss' and'reg_loss'.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = TensorDataset(data.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"recon_loss": [], "reg_loss": []}

    best_val = float("inf")
    patience = 0

    for epoch in range(epochs):
        recon_loss_sum = 0.0
        reg_loss_sum = 0.0

        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            recon_loss = loss_fn(recon, batch)
            reg_loss = model._regularizer(model.encode(batch))
            loss = recon_loss + reg_loss
            loss.backward()
            optimizer.step()

            recon_loss_sum += recon_loss.item() * batch.size(0)
            reg_loss_sum += reg_loss.item() * batch.size(0)

        recon_avg = recon_loss_sum / len(dataset)
        reg_avg = reg_loss_sum / len(dataset)
        history["recon_loss"].append(recon_avg)
        history["reg_loss"].append(reg_avg)

        if early_stop is not None and recon_avg < best_val:
            best_val = recon_avg
            patience = 0
        elif early_stop is not None:
            patience += 1
            if patience >= early_stop:
                break

    return history

__all__ = ["AutoencoderNet", "Autoencoder", "train_autoencoder"]
