import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

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

class FraudDetectionEnhanced(nn.Module):
    """Classical fraud detection network with residual blocks and Bayesian hyper‑parameter
    optimization support."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.input_params = input_params
        self.layers_params = list(layers)
        self.blocks = nn.ModuleList([self._build_block(p) for p in self.layers_params])
        self.output_layer = nn.Linear(2, 1)

    def _build_block(self, params: FraudLayerParameters) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.activation(self.linear(x))
                y = y * self.scale + self.shift
                return y + x  # residual connection

        return ResidualBlock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)

    def bayesian_hyperparameter_optimization(self, data_loader, num_trials: int = 50):
        """Placeholder Bayesian hyper‑parameter optimizer performing random search."""
        best_loss = float('inf')
        best_params = None
        for _ in range(num_trials):
            new_params = [FraudLayerParameters(
                bs_theta=param.bs_theta + 0.1 * torch.randn(1).item(),
                bs_phi=param.bs_phi + 0.1 * torch.randn(1).item(),
                phases=(param.phases[0] + 0.1 * torch.randn(1).item(),
                        param.phases[1] + 0.1 * torch.randn(1).item()),
                squeeze_r=(param.squeeze_r[0] + 0.1 * torch.randn(1).item(),
                           param.squeeze_r[1] + 0.1 * torch.randn(1).item()),
                squeeze_phi=(param.squeeze_phi[0] + 0.1 * torch.randn(1).item(),
                             param.squeeze_phi[1] + 0.1 * torch.randn(1).item()),
                displacement_r=(param.displacement_r[0] + 0.1 * torch.randn(1).item(),
                                param.displacement_r[1] + 0.1 * torch.randn(1).item()),
                displacement_phi=(param.displacement_phi[0] + 0.1 * torch.randn(1).item(),
                                  param.displacement_phi[1] + 0.1 * torch.randn(1).item()),
                kerr=(param.kerr[0] + 0.1 * torch.randn(1).item(),
                      param.kerr[1] + 0.1 * torch.randn(1).item()),
            ) for param in self.layers_params]
            model = FraudDetectionEnhanced(self.input_params, new_params)
            loss = self._evaluate_model(model, data_loader)
            if loss < best_loss:
                best_loss = loss
                best_params = new_params
        self.blocks = nn.ModuleList([self._build_block(p) for p in best_params])
        return best_loss

    def _evaluate_model(self, model, data_loader):
        model.eval()
        total_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            for x, y in data_loader:
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()
        return total_loss / len(data_loader)

__all__ = ["FraudDetectionEnhanced", "FraudLayerParameters"]
