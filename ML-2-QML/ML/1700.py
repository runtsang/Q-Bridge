import torch
from torch import nn
import numpy as np

class FullyConnectedLayer(nn.Module):
    """
    A fully‑connected neural layer that can be configured for any number of
    input and output features, optional bias, and a custom activation.
    The ``run`` method accepts a flat parameter vector and returns the
    layer's output for a unit input, mirroring the behaviour of the
    original seed while adding realism.
    """
    def __init__(
        self,
        n_features: int = 1,
        out_features: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.Tanh()
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, out_features, bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))

    def set_parameters(self, theta: np.ndarray) -> None:
        """
        Load a flat array of parameters into the layer's weights and bias.
        """
        param_array = torch.tensor(theta, dtype=torch.float32)
        idx = 0
        for param in self.parameters():
            num = param.numel()
            param.copy_(param_array[idx : idx + num].view_as(param))
            idx += num

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Accepts a flat vector of parameters, updates the layer, and
        returns the output for a unit input (shape [1, n_features]).
        """
        self.set_parameters(thetas)
        with torch.no_grad():
            inp = torch.ones((1, self.linear.in_features))
            out = self.forward(inp)
        return out.detach().cpu().numpy()
