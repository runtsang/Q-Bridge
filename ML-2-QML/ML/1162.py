"""Enhanced fully connected layer with multi‑layer support and training utilities.

The class can be instantiated with an arbitrary number of hidden layers,
custom activation functions, and a simple ``train`` method that uses
Adam to fit a regression target.  The ``run`` method accepts a list of
parameters that are reshaped into the weight matrices of the network,
mimicking the behaviour of the original seed but with richer
functionality.
"""

import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from typing import Iterable, Sequence, Callable, Optional


class FCL(nn.Module):
    """
    Multi‑layer fully‑connected neural network with flexible activation.

    Parameters
    ----------
    layer_sizes : Sequence[int]
        Sizes of the layers, e.g. ``(10, 20, 1)`` for a network with
        input size 10, one hidden layer of size 20, and a single output.
    activation : Callable[[torch.Tensor], torch.Tensor], optional
        Activation function applied after each hidden layer.  Defaults to
        ``torch.relu``.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least an input and an output size")
        self.activation = activation or torch.relu
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        out = x
        for i, layer in enumerate(self.model):
            out = layer(out)
            if i < len(self.model) - 1:
                out = self.activation(out)
        return out

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the network with a flattened list of parameters.

        The parameters are reshaped to match the weight matrices and
        biases of the underlying linear layers.  The method returns the
        mean output over a batch of size one.
        """
        # Flatten current parameters to compute size
        sizes = [param.numel() for param in self.parameters()]
        # Convert thetas to tensor and reshape
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32)
        if theta_tensor.numel()!= sum(sizes):
            raise ValueError("Number of thetas does not match network parameters")
        # Assign parameters
        idx = 0
        for param in self.parameters():
            num = param.numel()
            param.data = theta_tensor[idx : idx + num].view_as(param)
            idx += num
        # Forward pass on a dummy input
        dummy = torch.randn(1, self.model[0].in_features)
        out = self.forward(dummy)
        return out.detach().cpu().numpy()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        loss_fn: Callable = nn.MSELoss(),
    ) -> None:
        """
        Simple training loop using Adam optimiser.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        lr : float, optional
            Learning rate.
        epochs : int, optional
            Number of training epochs.
        batch_size : int, optional
            Batch size.
        loss_fn : Callable, optional
            Loss function, defaults to MSELoss.
        """
        device = torch.device("cpu")
        self.to(device)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.forward(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
