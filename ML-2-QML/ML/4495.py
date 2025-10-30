import torch
from torch import nn
import torch.nn.functional as F

class HybridAutoencoder(nn.Module):
    """
    Classical autoencoder that mirrors the structure of the quantum helper.
    It combines a convolutional encoder, a classical LSTM for temporal
    dependencies, a hybrid dense head, and a transposed‑convolutional decoder.
    """
    def __init__(self,
                 input_channels: int = 3,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.latent_linear = nn.Linear(64 * 4 * 4, latent_dim)
        self.lstm = nn.LSTM(latent_dim, lstm_hidden, lstm_layers, batch_first=True)
        self.hybrid_head = nn.Sequential(
            nn.Linear(lstm_hidden, 1),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.encoder(x)
        flat = self.flatten(enc)
        z = self.latent_linear(flat)
        z_seq = z.unsqueeze(1)
        lstm_out, _ = self.lstm(z_seq)
        head_out = self.hybrid_head(lstm_out.squeeze(1))
        dec = self.decoder(z)
        return dec, head_out

def train_hybrid_autoencoder(model: nn.Module,
                             data: torch.Tensor,
                             *,
                             epochs: int = 100,
                             batch_size: int = 64,
                             lr: float = 1e-3,
                             device: torch.device | None = None) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: torch.Tensor | list[float] | tuple[float,...]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["HybridAutoencoder", "train_hybrid_autoencoder"]
