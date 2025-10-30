import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionModel(nn.Module):
    """PyTorch implementation of the fraud detection architecture.

    This model is a drop‑in replacement for the original seed.  It adds
    optional dropout, a convenience ``from_parameters`` constructor,
    and a ``fit`` method that uses Adam to train the network on a
    binary classification task.  The final linear layer outputs a
    single logit that can be passed through a sigmoid during training.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([self._layer_from_params(input_params, clip=False)])
        self.layers.extend(self._layer_from_params(layer, clip=True) for layer in layers)
        self.layers.append(nn.Linear(2, 1))
        self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()

    def _layer_from_params(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                out = self.activation(self.linear(x))
                out = out * self.scale + self.shift
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.dropout(layer(x))
        return self.layers[-1](x)

    @classmethod
    def from_parameters(
        cls,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float | None = None,
    ) -> "FraudDetectionModel":
        """Convenience constructor that accepts the parameter dataclasses directly."""
        return cls(input_params, layers, dropout=dropout)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        """Train the model using Adam and binary cross‑entropy loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            logits = self.forward(X).squeeze()
            loss = criterion(logits, y.float())
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs:3d}  loss={loss.item():.4f}")

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
