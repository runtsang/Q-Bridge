import torch
from torch import nn
import numpy as np

class FCL(nn.Module):
    """
    Classical hybrid fully‑connected layer that mimics the behaviour of the
    original FCL example while exposing a sampler‑style interface.

    The module contains two sub‑networks:
        * ``self.linear`` – a single linear transform followed by tanh,
          producing an expectation‑like scalar.
        * ``self.sampler_net`` – a small network that maps the same
          input to four parameters which can be fed into a quantum sampler.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.sampler_net = nn.Sequential(
            nn.Linear(n_features, 4),
            nn.Tanh(),
            nn.Linear(4, 4)
        )

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Return the expectation‑like value computed by the linear branch.
        """
        expectation = torch.tanh(self.linear(thetas)).mean(dim=0)
        return expectation

    def get_sampler_params(self, thetas: torch.Tensor) -> np.ndarray:
        """
        Produce a tensor of four parameters that can be used by a quantum
        sampler.  The values are returned as a NumPy array for easy
        interoperability with the QML implementation.
        """
        params = self.sampler_net(thetas)
        return params.detach().cpu().numpy()

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Compatibility wrapper that simply forwards to ``forward``.
        """
        return self.forward(thetas)

__all__ = ["FCL"]
