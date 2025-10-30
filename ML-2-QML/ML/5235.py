import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Iterable, Tuple

# ------------------------------------------------------------------
# Classical autoencoder with optional downstream heads
# ------------------------------------------------------------------
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int,...] = (128, 64),
                 dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h
        encoder.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)]
            in_dim = h
        decoder.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder)

        # optional downstream heads
        self.reg_head = nn.Linear(cfg.latent_dim, 1)
        self.cls_head = nn.Linear(cfg.latent_dim, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def predict_reg(self, z: torch.Tensor) -> torch.Tensor:
        return self.reg_head(z)

    def predict_cls(self, z: torch.Tensor) -> torch.Tensor:
        return self.cls_head(z)

def train_autoencoder(model: AutoencoderNet,
                      data: torch.Tensor,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      device: torch.device | None = None) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_f = nn.MSELoss()
    history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch,_ in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_f(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history

# ------------------------------------------------------------------
# Dataset helpers (classical regression)
# ------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.X, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return {"states": torch.tensor(self.X[idx], dtype=torch.float32),
                "target": torch.tensor(self.y[idx], dtype=torch.float32)}

# ------------------------------------------------------------------
# Fast estimators for classical models
# ------------------------------------------------------------------
class FastBaseEstimator:
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Iterable[Iterable[float]]) -> list[list[float]]:
        obs = list(observables) or [lambda o: o.mean(dim=-1)]
        results = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                out = self.model(inp)
                row = []
                for f in obs:
                    val = f(out)
                    row.append(float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(self,
                 observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
                 parameter_sets: Iterable[Iterable[float]],
                 shots: int | None = None,
                 seed: int | None = None) -> list[list[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(m, max(1e-6, 1 / shots))) for m in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["AutoencoderConfig", "AutoencoderNet", "train_autoencoder",
           "generate_superposition_data", "RegressionDataset",
           "FastBaseEstimator", "FastEstimator"]
