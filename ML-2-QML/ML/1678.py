import numpy as np
import torch
from torch import nn
from typing import Iterable, List

class FCL(nn.Module):
    """
    Classical block‑structured fully‑connected layer.
    Supports a single linear map and optional bias.  The
    ``run`` method accepts a flat list of parameters that
    are written directly into the underlying ``nn.Linear``
    instance, mimicking the behaviour of the original seed.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Apply a parameter vector to the linear layer and
        return the output on a unit‑input vector.

        Parameters
        ----------
        thetas : iterable of float
            Flat list containing first the weights
            (in column‑major order) followed by the bias
            (if present).  Length must match the number
            of trainable parameters in ``self.linear``.
        """
        param_tensor = torch.tensor(list(thetas), dtype=torch.float32)
        weight_num = self.linear.weight.numel()
        bias_num = self.linear.bias.numel() if self.linear.bias is not None else 0
        if param_tensor.numel()!= weight_num + bias_num:
            raise ValueError("Length of ``thetas`` does not match layer parameters.")
        with torch.no_grad():
            self.linear.weight.copy_(
                param_tensor[:weight_num].reshape(self.linear.weight.shape)
            )
            if bias_num:
                self.linear.bias.copy_(
                    param_tensor[weight_num:].reshape(self.linear.bias.shape)
                )
        # Dummy input of ones to emulate the reference behaviour
        dummy = torch.ones((1, self.linear.in_features))
        out = self.forward(dummy)
        return out.detach().numpy()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> "FCL":
        """
        Train the linear layer on simple regression data
        using mean‑squared error loss.

        Parameters
        ----------
        X, y : np.ndarray
            Training data.
        epochs : int
            Number of gradient descent steps.
        lr : float
            Learning rate for the Adam optimiser.
        verbose : bool
            If True, prints loss every 10 epochs.
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            pred = self.forward(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d} – loss: {loss.item():.4f}")
        return self

__all__ = ["FCL"]
