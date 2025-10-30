import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

# Import the quantum decoder helper
from.quantum_decoder import build_quantum_decoder


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    device: torch.device | None = None


class HybridAutoencoder(nn.Module):
    """A hybrid MLP–quantum autoencoder.

    The encoder is a standard feed‑forward network.  The decoder is a
    variational quantum circuit implemented with Pennylane.  The
    network can be trained end‑to‑end with a gradient‑based optimiser.
    """
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ---------- Encoder -------------------------------------------------- #
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ---------- Quantum decoder ------------------------------------------ #
        self.decoder_qnode, self.decoder_params = build_quantum_decoder(
            cfg.latent_dim, cfg.input_dim
        )
        # Register quantum parameters as nn.ParameterList for optimizer
        self.decoder_params = nn.ParameterList([nn.Parameter(p) for p in self.decoder_params])

    # ----------------------------------------------------------------------- #
    #  Forward pass
    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode + decode.  The decoder returns a torch tensor on the same
        device as the input."""
        latent = self.encoder(x.to(self.device))
        return self.decoder_qnode(latent)

    # ----------------------------------------------------------------------- #
    #  Helpers
    # ----------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(x.to(self.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector via the quantum circuit."""
        return self.decoder_qnode(z)

    def latent_clustering(self, data: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Cluster the latent space using K‑Means and return cluster labels."""
        from sklearn.cluster import KMeans

        z = self.encode(data).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=k, random_state=42)
        return torch.tensor(kmeans.fit_predict(z), dtype=torch.long)

# --------------------------------------------------------------------------- #
#  Training routine
# --------------------------------------------------------------------------- #
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid model end‑to‑end.

    The routine uses Adam and MSE loss.  The quantum decoder
    back‑propagates through the Pennylane QNode.
    """
    device = device or model.device
    model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder_params), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
#  Utility
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = ["HybridAutoencoder", "AutoencoderConfig", "train_hybrid_autoencoder"]
