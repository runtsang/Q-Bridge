import torch
from torch import nn
import torch.utils.data
from dataclasses import dataclass
from typing import Tuple, Iterable, Sequence, Callable, List
import numpy as np

# --------------------------------------------------------------------------- #
#  Classical autoencoder definition
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderHybridConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class AutoencoderHybridNet(nn.Module):
    """Fully‑connected autoencoder with configurable hidden layers."""
    def __init__(self, cfg: AutoencoderHybridConfig):
        super().__init__()
        self.encoder = self._make_mlp(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.dropout)
        self.decoder = self._make_mlp(cfg.latent_dim, tuple(reversed(cfg.hidden_dims)), cfg.input_dim, cfg.dropout)

    def _make_mlp(self, in_dim, hidden_dims, out_dim, dropout):
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def AutoencoderHybrid(input_dim: int, **kwargs) -> AutoencoderHybridNet:
    cfg = AutoencoderHybridConfig(input_dim=input_dim, **kwargs)
    return AutoencoderHybridNet(cfg)

# --------------------------------------------------------------------------- #
#  Training loop
# --------------------------------------------------------------------------- #
def train_autoencoder_hybrid(
    model: nn.Module,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

# --------------------------------------------------------------------------- #
#  Fast estimator (deterministic or shot‑noisy)
# --------------------------------------------------------------------------- #
class FastEstimator:
    """Evaluate a torch model on many parameter sets, with optional Gaussian noise."""
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        obs = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.model(inp)
                row = [float(f(out).mean().cpu()) for f in obs]
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [
            [float(rng.normal(m, max(1e-6, 1 / shots))) for m in row]
            for row in results
        ]
        return noisy

__all__ = [
    "AutoencoderHybridConfig",
    "AutoencoderHybridNet",
    "AutoencoderHybrid",
    "train_autoencoder_hybrid",
    "FastEstimator",
]
