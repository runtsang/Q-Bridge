from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_quantum_layer: bool = False
    quantum_layer_cls: type | None = None

# --------------------------------------------------------------------------- #
#  Hybrid auto‑encoder
# --------------------------------------------------------------------------- #
class HybridAutoencoderQCL(nn.Module):
    """Classical MLP auto‑encoder optionally followed by a quantum fully‑connected layer."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(*self._build_mlp(config.input_dim,
                                                      config.hidden_dims,
                                                      config.latent_dim))
        self.decoder = nn.Sequential(*self._build_mlp(config.latent_dim,
                                                      tuple(reversed(config.hidden_dims)),
                                                      config.input_dim))
        self.quantum_layer = None
        if config.use_quantum_layer and config.quantum_layer_cls is not None:
            self.quantum_layer = config.quantum_layer_cls()

    def _build_mlp(self, in_dim: int, hidden_dims: Tuple[int,...], out_dim: int):
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return layers

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantum_encode(self, z: torch.Tensor) -> torch.Tensor:
        if self.quantum_layer is None:
            raise RuntimeError("Quantum layer not configured")
        thetas = z.detach().cpu().numpy().flatten()
        out = torch.as_tensor(self.quantum_layer.run(thetas), dtype=torch.float32)
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.quantum_layer is not None:
            z = self.quantum_encode(z)
        return self.decode(z)

# --------------------------------------------------------------------------- #
#  Factory (mirrors quantum helper)
# --------------------------------------------------------------------------- #
def Autoencoder(input_dim: int,
                *,
                latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1,
                use_quantum_layer: bool = False,
                quantum_layer_cls: type | None = None) -> HybridAutoencoderQCL:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quantum_layer=use_quantum_layer,
        quantum_layer_cls=quantum_layer_cls,
    )
    return HybridAutoencoderQCL(cfg)

# --------------------------------------------------------------------------- #
#  Training loop
# --------------------------------------------------------------------------- #
def train_autoencoder(model: HybridAutoencoderQCL,
                      data: torch.Tensor,
                      *,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# --------------------------------------------------------------------------- #
#  Hybrid classifier mirroring quantum helper
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int,
                             depth: int) -> Tuple[nn.Sequential, Sequence[int], Sequence[int], List[int]]:
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.extend([lin, nn.ReLU()])
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    net = nn.Sequential(*layers)
    observables = list(range(2))
    return net, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
#  Classical Quanvolution
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feat = self.qfilter(x)
        logits = self.linear(feat)
        return torch.nn.functional.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------------- #
#  Fully connected quantum layer placeholder
# --------------------------------------------------------------------------- #
def FCL() -> nn.Module:
    """Return a simple class mimicking the quantum fully‑connected layer."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().cpu().numpy()

    return FullyConnectedLayer()
