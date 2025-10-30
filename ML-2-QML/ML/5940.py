import torch
from torch import nn
from torch.distributions import Bernoulli
from dataclasses import dataclass
from typing import Iterable, Tuple, List

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

class FraudDetectionAdv:
    '''Classical fraud detection model with Bayesian calibration.'''

    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 device: str = 'cpu'):
        self.device = device
        self.input_params = input_params
        self.layers = list(layers)
        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        modules: List[nn.Module] = [self._layer_from_params(self.input_params, clip=False)]
        modules += [self._layer_from_params(l, clip=True) for l in self.layers]
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules).to(self.device)

    def _layer_from_params(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32, device=self.device)
        bias = torch.tensor(params.phases, dtype=torch.float32, device=self.device)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2).to(self.device)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        activation = nn.Tanh()
        scale = torch.tensor(params.displacement_r, dtype=torch.float32, device=self.device)
        shift = torch.tensor(params.displacement_phi, dtype=torch.float32, device=self.device)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                outputs = self.activation(self.linear(inputs))
                outputs = outputs * self.scale + self.shift
                return outputs

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Return calibrated Bernoulli sample given input features.'''
        logits = self.model(x.to(self.device))
        probs = torch.sigmoid(logits)
        return Bernoulli(probs).sample()

    def get_parameters(self):
        return [p.detach().cpu() for p in self.model.parameters()]

    def set_parameters(self, param_list):
        for p, new in zip(self.model.parameters(), param_list):
            p.data.copy_(new.to(self.device))

__all__ = ["FraudDetectionAdv", "FraudLayerParameters"]
