import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input data to a float32 torch.Tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class SamplerQNN(nn.Module):
    """Classical sampler network mirroring the quantum SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class EstimatorQNN(nn.Module):
    """Classical regression network mirroring the quantum EstimatorQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class ClassifierNet(nn.Module):
    """Feed‑forward classifier that mimics build_classifier_circuit."""
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class AutoencoderHybrid(nn.Module):
    """Hybrid autoencoder that can embed quantum sampler/estimator modules."""
    def __init__(self,
                 config: AutoencoderConfig,
                 quantum_sampler: Optional[nn.Module] = None,
                 quantum_estimator: Optional[nn.Module] = None,
                 classifier: Optional[nn.Module] = None) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Optional quantum modules
        self.quantum_sampler = quantum_sampler
        self.quantum_estimator = quantum_estimator
        self.classifier = classifier

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        if self.quantum_sampler is not None:
            probs = self.quantum_sampler(latent)
            latent = torch.matmul(probs, torch.arange(probs.size(-1), device=probs.device))
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.quantum_estimator is not None:
            return self.quantum_estimator(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("No classifier module attached.")
        return self.classifier(x)

def AutoencoderHybridFactory(input_dim: int,
                             *,
                             latent_dim: int = 32,
                             hidden_dims: Tuple[int, int] = (128, 64),
                             dropout: float = 0.1,
                             quantum_sampler: Optional[nn.Module] = None,
                             quantum_estimator: Optional[nn.Module] = None,
                             classifier: Optional[nn.Module] = None) -> AutoencoderHybrid:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderHybrid(cfg,
                             quantum_sampler=quantum_sampler,
                             quantum_estimator=quantum_estimator,
                             classifier=classifier)

def train_autoencoder(model: nn.Module,
                      data: torch.Tensor,
                      *,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

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

def train_classifier(classifier: nn.Module,
                     data: torch.Tensor,
                     labels: torch.Tensor,
                     *,
                     epochs: int = 50,
                     batch_size: int = 32,
                     lr: float = 1e-3,
                     weight_decay: float = 0.0,
                     device: torch.device | None = None) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data), _as_tensor(labels))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch, lab) in loader:
            batch, lab = batch.to(device), lab.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch)
            loss = loss_fn(logits, lab.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridFactory",
    "AutoencoderConfig",
    "SamplerQNN",
    "EstimatorQNN",
    "ClassifierNet",
    "train_autoencoder",
    "train_classifier",
]
