import torch
import torch.nn as nn
from typing import Iterable, Optional

class FCL(nn.Module):
    """
    Classical fully‑connected layer with optional dropout and a mixed‑precision
    training helper. The class can be instantiated with a specific number of
    input features and an optional dropout probability.

    Example:
        >>> model = FCL(n_features=1, dropout=0.3)
        >>> output = model.run([0.1, 0.2, 0.3])
    """

    def __init__(self, n_features: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return torch.tanh(self.linear(x))

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Compute the mean activation for a batch of input values.

        Parameters
        ----------
        thetas : Iterable[float]
            Collection of input values to feed through the layer.

        Returns
        -------
        torch.Tensor
            Mean output of the layer over the input batch.
        """
        values = torch.tensor(list(thetas), dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            out = self.forward(values).mean()
        return out

    def train_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn,
        optimizer,
        amp: Optional[nn.Module] = None,
    ) -> float:
        """
        Demonstrates a mixed‑precision training step.

        Parameters
        ----------
        inputs : torch.Tensor
            Batch of input samples.
        targets : torch.Tensor
            Ground‑truth targets.
        loss_fn : callable
            Loss function.
        optimizer : torch.optim.Optimizer
            Optimizer to update parameters.
        amp : torch.cuda.amp.GradScaler, optional
            Gradient scaler for mixed‑precision training.

        Returns
        -------
        float
            Scalar loss value for the batch.
        """
        optimizer.zero_grad()
        with torch.autocast(device_type='cpu'):
            preds = self.forward(inputs)
            loss = loss_fn(preds, targets)
        if amp is not None:
            amp.scale(loss).backward()
            amp.step(optimizer)
            amp.update()
        else:
            loss.backward()
            optimizer.step()
        return loss.item()

__all__ = ["FCL"]
