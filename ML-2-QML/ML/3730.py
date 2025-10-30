import torch
from torch import nn
import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Iterable

# ---------- Classical QCNN‑inspired block ----------
class QCNNModel(nn.Module):
    """Convolution‑like feed‑forward block mirroring the QCNN helper."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# ---------- Configuration ----------
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_qcnn: bool = False  # whether to prepend QCNN block

# ---------- Quantum layer ----------
def _get_qnode(num_qubits: int, weights: torch.Tensor):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # encode input as rotations
        for i in range(num_qubits):
            qml.RY(x[i], wires=i)
        # variational ansatz
        qml.templates.BasicEntanglerLayers(w, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return qnode

# ---------- Hybrid Autoencoder ----------
class HybridAutoencoder(nn.Module):
    """Hybrid classical‑quantum autoencoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # classical encoder
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

        # optional QCNN feature extractor
        self.use_qcnn = config.use_qcnn
        if self.use_qcnn:
            self.qcnn = QCNNModel()

        # quantum variational layer
        self.quantum_weights = nn.Parameter(
            torch.randn(config.latent_dim, 3)
        )  # BasicEntanglerLayers needs num_qubits x 3 parameters
        self.qnode = _get_qnode(config.latent_dim, self.quantum_weights)

        # classical decoder
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

    # ---------- helpers ----------
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.encoder(inputs)
        if self.use_qcnn:
            x = self.qcnn(x)
        return x

    def quantum_transform(self, latent: torch.Tensor) -> torch.Tensor:
        # ensure latent is one‑dim per sample
        return self.qnode(latent, self.quantum_weights)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        q_latent = self.quantum_transform(latent)
        return self.decode(q_latent)

# ---------- factory ----------
def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_qcnn: bool = False,
) -> HybridAutoencoder:
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_qcnn=use_qcnn,
    )
    return HybridAutoencoder(config)

# ---------- training loop ----------
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(_as_tensor(data)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataloader.dataset)
        history.append(epoch_loss)
    return history

# ---------- utilities ----------
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
