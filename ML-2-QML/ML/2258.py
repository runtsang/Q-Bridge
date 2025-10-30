import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

__all__ = ["UnifiedAutoencoder", "UnifiedAutoencoderConfig"]

class UnifiedAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int,...] = (128, 64),
                 dropout: float = 0.0,
                 use_quantum: bool = False,
                 quantum_latent_dim: int = 3,
                 quantum_trash_dim: int = 2):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_quantum = use_quantum
        self.quantum_latent_dim = quantum_latent_dim
        self.quantum_trash_dim = quantum_trash_dim

class UnifiedAutoencoder(nn.Module):
    """Hybrid autoencoder that can operate in pure‑classical or hybrid‑quantum mode."""
    def __init__(self, config: UnifiedAutoencoderConfig):
        super().__init__()
        self.config = config
        # Classical encoder
        enc_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.classical_encoder = nn.Sequential(*enc_layers)

        # Classical decoder
        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.classical_decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the classical latent representation."""
        return self.classical_encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent representation."""
        return self.classical_decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        return self.decode(self.encode(x))

    def hybrid_encode(self, x: torch.Tensor, quantum_encoder) -> torch.Tensor:
        """
        Encode using a supplied quantum encoder function.
        `quantum_encoder` should accept a torch tensor and return a tensor of shape
        (batch, quantum_latent_dim).
        """
        if not self.config.use_quantum:
            raise RuntimeError("Hybrid mode disabled in config.")
        return quantum_encoder(x)

    def hybrid_decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode quantum latent representation using the classical decoder."""
        return self.decode(z)

    def train_autoencoder(self,
                          data: torch.Tensor | np.ndarray,
                          *,
                          epochs: int = 100,
                          batch_size: int = 64,
                          lr: float = 1e-3,
                          weight_decay: float = 0.0,
                          device: torch.device | None = None) -> list[float]:
        """Simple reconstruction training loop returning the loss history."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(self._as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                reconstruction = self(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    @staticmethod
    def _as_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(dtype=torch.float32)
        return torch.as_tensor(data, dtype=torch.float32)
