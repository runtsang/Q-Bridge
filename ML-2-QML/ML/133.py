import numpy as np
import torch
from torch import nn
from torch.optim import Adam

class FCL(nn.Module):
    """
    Fully‑connected linear layer with training capability.

    The class implements a simple feed‑forward network consisting of a single
    linear transformation followed by a tanh activation.  It exposes a `run`
    method that mimics the original seed, a `train_step` that performs a
    single optimisation step and a `train` helper that iterates over a
    dataset for a number of epochs.
    """

    def __init__(self, n_features: int = 1, n_outputs: int = 1, lr: float = 1e-3):
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)
        self.loss_fn = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Mimic the original interface.  ``thetas`` is interpreted as a
        sequence of input values to the linear layer.
        """
        x = torch.as_tensor(thetas, dtype=torch.float32).view(-1, self.linear.in_features)
        out = self.forward(x)
        return out.detach().numpy()

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Perform a single optimisation step.

        Parameters
        ----------
        x : np.ndarray
            Input batch.
        y : np.ndarray
            Target values.

        Returns
        -------
        loss : float
            The scalar loss value after the step.
        """
        self.train()
        x_t = torch.as_tensor(x, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        self.optimizer.zero_grad()
        pred = self.forward(x_t)
        loss = self.loss_fn(pred, y_t)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, dataset, epochs: int = 10):
        """
        Simple training loop over an iterable of (x, y) pairs.

        Parameters
        ----------
        dataset : Iterable[Tuple[np.ndarray, np.ndarray]]
            Iterable yielding input/target pairs.
        epochs : int
            Number of epochs to run.
        """
        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0
            for x, y in dataset:
                loss = self.train_step(x, y)
                epoch_loss += loss
                count += 1
            print(f"Epoch {epoch+1}/{epochs}  Loss: {epoch_loss/count:.4f}")
