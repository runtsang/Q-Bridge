import torch
from torch import nn, optim
from typing import List

class QCNNModel(nn.Module):
    """
    Classical QCNN-inspired network with residual connections and dropout.
    Extends the original 8→16→...→1 architecture by adding skip connections,
    configurable hidden dimensions and dropout for better regularisation.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] | None = None,
        activation: nn.Module = nn.Tanh(),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), activation, nn.Dropout(dropout)])
            prev_dim = dim
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)
        # residual connection from input to final hidden layer
        if input_dim == prev_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, prev_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat + self.residual(x)
        return torch.sigmoid(self.head(feat))

    def freeze(self, names: List[str] | None = None) -> None:
        """
        Freeze layers whose names appear in *names*.
        """
        if names is None:
            return
        for name, param in self.named_parameters():
            if any(name.startswith(n) for n in names):
                param.requires_grad = False

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        optimizer.zero_grad()
        output = self(batch["x"])
        loss = criterion(output, batch["y"])
        loss.backward()
        optimizer.step()
        return loss.item()

def QCNN() -> QCNNModel:
    """
    Factory returning a configured QCNNModel instance.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
