"""PyTorch implementation of a convolutional autoencoder with early stopping."""
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class AutoencoderConfig:
    input_channels: int
    latent_dim: int = 64
    hidden_dims: Tuple[int,...] = (32, 16)
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    dropout: float = 0.1
    batch_norm: bool = True

class AutoencoderModel(nn.Module):
    """Convolutional autoencoder with configurable architecture."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Encoder
        enc_layers = []
        in_ch = config.input_channels
        for out_ch in config.hidden_dims:
            enc_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=config.kernel_size,
                                        stride=config.stride, padding=config.padding))
            if config.batch_norm:
                enc_layers.append(nn.BatchNorm2d(out_ch))
            enc_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0:
                enc_layers.append(nn.Dropout2d(config.dropout))
            in_ch = out_ch
        enc_layers.append(nn.Flatten())
        # Linear bottleneck
        dummy = torch.zeros(1, config.input_channels, 32, 32)
        with torch.no_grad():
            x = dummy
            for layer in enc_layers[:-1]:
                x = layer(x)
        flat_dim = x.shape[1]
        self.encoder = nn.Sequential(*enc_layers)
        self.bottleneck = nn.Linear(flat_dim, config.latent_dim)
        # Decoder
        dec_layers = [nn.Linear(config.latent_dim, flat_dim)]
        dec_layers.append(nn.Unflatten(1, x.shape[1:]))
        in_ch = flat_dim
        hidden_dims_rev = config.hidden_dims[::-1]
        for out_ch in hidden_dims_rev:
            dec_layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=config.kernel_size,
                                                 stride=config.stride, padding=config.padding,
                                                 output_padding=1))
            if config.batch_norm:
                dec_layers.append(nn.BatchNorm2d(out_ch))
            dec_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0:
                dec_layers.append(nn.Dropout2d(config.dropout))
            in_ch = out_ch
        dec_layers.append(nn.ConvTranspose2d(in_ch, config.input_channels,
                                             kernel_size=config.kernel_size,
                                             stride=config.stride, padding=config.padding,
                                             output_padding=1))
        dec_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def create_autoencoder(config: AutoencoderConfig) -> AutoencoderModel:
    """Factory that returns a configured autoencoder."""
    return AutoencoderModel(config)

def train_autoencoder(
    model: AutoencoderModel,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.1,
    patience: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[List[float], List[float]]:
    """Train with early stopping. Returns train and validation loss histories."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
    train_len = int(len(dataset) * (1 - val_split))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    train_hist, val_hist = [], []
    best_val = float("inf")
    counter = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_hist.append(epoch_loss)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    return train_hist, val_hist

__all__ = ["AutoencoderModel", "AutoencoderConfig", "create_autoencoder", "train_autoencoder"]
