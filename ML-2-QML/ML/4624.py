import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Classical QCNN feature extractor (from the QCNN reference)
class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# Classical radial‑basis kernel (from the QuantumKernelMethod reference)
class KernalAnsatz(nn.Module):
    """Placeholder maintaining compatibility with the quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# Hybrid autoencoder that uses QCNN features and a quantum‑kernel regularizer
class AutoencoderGen066Net(nn.Module):
    """Classical autoencoder that feeds QCNN features into a standard MLP."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        # Map raw data to 8‑dimensional QCNN input
        self.pre_qcnn = nn.Linear(input_dim, 8)
        self.qcnn = QCNNModel()
        # Encoder
        encoder_layers = []
        in_dim = 1  # QCNN returns a single feature
        for h in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pre_qcnn(x)
        qcnn_feat = self.qcnn(x)
        z = self.encoder(qcnn_feat)
        recon = self.decoder(z)
        return recon, z

def autoencoder_gen066(input_dim: int,
                       latent_dim: int = 32,
                       hidden_dims: tuple[int, int] = (128, 64),
                       dropout: float = 0.1) -> AutoencoderGen066Net:
    """Factory that returns the hybrid autoencoder."""
    return AutoencoderGen066Net(input_dim, latent_dim, hidden_dims, dropout)

def train_autoencoder_gen066(net: AutoencoderGen066Net,
                             data: torch.Tensor,
                             *,
                             epochs: int = 100,
                             batch_size: int = 64,
                             lr: float = 1e-3,
                             weight_decay: float = 0.0,
                             device: torch.device | None = None) -> list[float]:
    """Training loop that adds a quantum‑kernel regularizer on the latent space."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    dataset = TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    kernel = Kernel(gamma=0.5)
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            recon, z = net(batch)
            loss = mse(recon, batch)
            # Regularize latent space with a quantum kernel similarity term
            loss += 0.01 * torch.mean(kernel(z, z))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * batch.size(0)
        history.append(epoch_loss / len(dataset))
    return history

__all__ = [
    "AutoencoderGen066Net",
    "autoencoder_gen066",
    "train_autoencoder_gen066",
]
