import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    A configurable classical sampler network.
    Extends the original two‑layer network by supporting:
        • multiple hidden layers with user‑defined widths,
        • optional batch‑normalisation,
        • dropout for regularisation,
        • a convenience train_step method for quick experiments.
    The network outputs a probability vector via softmax.
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dims: tuple[int,...] = (8, 8),
                 output_dim: int = 2,
                 dropout: float = 0.1,
                 batch_norm: bool = True,
                 activation: nn.Module = nn.Tanh()) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability distribution over the output classes."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Convenience helpers for quick experiments
    # ------------------------------------------------------------------
    def loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross‑entropy loss against one‑hot targets."""
        return F.cross_entropy(preds, targets.argmax(dim=-1))

    def train_step(self,
                   optimizer: torch.optim.Optimizer,
                   data: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        """Perform a single gradient update step."""
        self.train()
        optimizer.zero_grad()
        preds = self(data)
        loss = self.loss_fn(preds, target)
        loss.backward()
        optimizer.step()
        return loss

    def evaluate(self, data: torch.Tensor) -> torch.Tensor:
        """Return argmax predictions for the given data."""
        self.eval()
        with torch.no_grad():
            probs = self(data)
        return probs.argmax(dim=-1)

__all__ = ["SamplerQNN"]
