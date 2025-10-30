import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple
from sklearn.preprocessing import StandardScaler

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _build_photonic_layer(params: FraudLayerParameters, clip: bool) -> nn.Module:
    # Construct a weight matrix that encodes the photonic beam‑splitter
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = torch.clamp(weight, -5.0, 5.0)
        bias = torch.clamp(bias, -5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    class ScaleShift(nn.Module):
        def __init__(self, scale, shift):
            super().__init__()
            self.scale = nn.Parameter(scale, requires_grad=False)
            self.shift = nn.Parameter(shift, requires_grad=False)

        def forward(self, x):
            return x * self.scale + self.shift

    return nn.Sequential(
        linear,
        nn.Tanh(),
        ScaleShift(
            torch.tensor(params.displacement_r, dtype=torch.float32),
            torch.tensor(params.displacement_phi, dtype=torch.float32),
        ),
    )

class FraudDetectionHybrid(nn.Module):
    """Hybrid model that emulates a photonic circuit with a classical NN."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self.layers = nn.ModuleList([_build_photonic_layer(input_params, clip=False)])
        self.layers.extend(_build_photonic_layer(layer, clip=True) for layer in layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.final = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        return self.final(x)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        X_scaled = self.scaler.fit_transform(X)
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y.reshape(-1, 1), dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if verbose and epoch % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"[Epoch {epoch}] loss={avg_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            logits = self(torch.tensor(X_scaled, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
        return (probs > 0.5).astype(int).reshape(-1)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
