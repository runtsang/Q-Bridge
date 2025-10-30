import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Optional, Sequence, Callable, Union


class HybridFCL(nn.Module):
    """
    A hybrid neural network that stacks classical dense layers with a
    quantum‑parameterized layer. The classical part supports multiple
    hidden units, dropout, and L2 regularization. The quantum part
    is represented by a callable that returns expectation values
    given parameter tensors. By default the quantum contribution
    is a no‑op that returns zeros, but users can inject a real
    quantum circuit.

    Args:
        n_features (int): Number of input features.
        hidden_units (Sequence[int] | None): Sizes of hidden dense layers.
        dropout (float): Dropout probability applied after each hidden layer.
        l2 (float): L2 regularization weight on dense layer weights.
        device (torch.device | str): Device on which to run the network.
        quantum_layer (Callable[[torch.Tensor], torch.Tensor] | None):
            Optional callable that accepts a tensor of parameters and
            returns a tensor of quantum expectation values.
    """
    def __init__(
        self,
        n_features: int = 1,
        hidden_units: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        l2: float = 1e-4,
        device: Union[torch.device, str] = "cpu",
        quantum_layer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.device = torch.device(device)
        hidden_units = hidden_units or []
        layers = []
        input_dim = n_features
        for h in hidden_units:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))  # output layer
        self.classical = nn.Sequential(*layers).to(self.device)
        self.l2 = l2
        self.quantum_layer = quantum_layer or self._default_quantum_layer

    def _default_quantum_layer(self, thetas: torch.Tensor) -> torch.Tensor:
        """Fallback quantum layer that returns zeros."""
        return torch.zeros(thetas.shape[0], 1, device=self.device)

    def forward(self, x: torch.Tensor, thetas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch, n_features).
        thetas : torch.Tensor | None
            Quantum parameters of shape (batch, n_params).  If None, the
            quantum contribution is omitted.

        Returns
        -------
        torch.Tensor
            Output of shape (batch, 1) combining classical and quantum
            predictions.
        """
        x = x.to(self.device)
        class_out = self.classical(x)
        if thetas is not None:
            thetas = thetas.to(self.device)
            quantum_out = self.quantum_layer(thetas)
            return class_out + quantum_out
        return class_out

    def l2_regularization(self) -> torch.Tensor:
        """Compute L2 penalty over all trainable weights."""
        l2 = 0.0
        for param in self.classical.parameters():
            l2 += torch.norm(param, 2) ** 2
        return self.l2 * l2

    def sample_quantum_parameters(
        self,
        num_samples: int,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> torch.Tensor:
        """
        Draw samples from a normal prior over the quantum parameters
        for Bayesian inference.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        mean : float
            Mean of the prior.
        std : float
            Standard deviation of the prior.

        Returns
        -------
        torch.Tensor
            Samples of shape (num_samples, n_params).
        """
        # Infer number of parameters from quantum_layer signature
        n_params = 1
        if hasattr(self.quantum_layer, "__code__"):
            n_params = max(1, len(self.quantum_layer.__code__.co_varnames) - 1)
        return torch.randn(num_samples, n_params, device=self.device) * std + mean


__all__ = ["HybridFCL"]
