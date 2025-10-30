import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """
    A flexible fully‑connected regressor with optional dropout and batch
    normalisation.  The architecture can be tuned via ``hidden_dims`` and
    ``dropout`` to accommodate a wider range of regression tasks.
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: list[int] | tuple[int,...] = (8, 4),
                 dropout: float | None = None) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_step(self,
                   data_loader,
                   optimizer,
                   loss_fn=nn.MSELoss(),
                   device: str | torch.device = 'cpu') -> float:
        """
        Execute one training epoch and return the mean loss.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Batch iterator yielding (inputs, targets).
        optimizer : torch.optim.Optimizer
            Optimiser used to update the network parameters.
        loss_fn : torch.nn.Module, optional
            Loss function; defaults to MSELoss.
        device : str or torch.device, optional
            Device to perform computations on.
        """
        self.train()
        total_loss = 0.0
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = self(X).squeeze()
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
        return total_loss / len(data_loader.dataset)

    def predict(self,
                X: torch.Tensor,
                device: str | torch.device = 'cpu') -> torch.Tensor:
        """
        Predict on a batch of inputs without gradients.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (N, input_dim).
        device : str or torch.device, optional
            Device to perform inference on.
        """
        self.eval()
        with torch.no_grad():
            return self(X.to(device)).cpu()
__all__ = ["EstimatorQNN"]
