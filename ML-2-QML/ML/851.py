"""Enhanced QCNN model with residual connections, dropout, and a PyTorch Lightning training wrapper."""

import torch
from torch import nn
import pytorch_lightning as pl

class QCNNModel(nn.Module):
    """A deep classical QCNN-inspired network with residual connections and dropout."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        assert len(hidden_dims) % 2 == 0, "hidden_dims should contain an even number of layers (conv+pool pairs)."

        layers = []
        in_dim = input_dim
        self.residuals = nn.ModuleList()

        for i in range(0, len(hidden_dims), 2):
            conv_dim = hidden_dims[i]
            pool_dim = hidden_dims[i + 1]

            conv = nn.Sequential(
                nn.Linear(in_dim, conv_dim),
                nn.BatchNorm1d(conv_dim),
                nn.Tanh(),
            )
            pool = nn.Sequential(
                nn.Linear(conv_dim, pool_dim),
                nn.BatchNorm1d(pool_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
            )
            layers.append(conv)
            layers.append(pool)

            # Residual connection from input of this pair to output of pooling
            self.residuals.append(
                nn.Sequential(
                    nn.Linear(in_dim, pool_dim),
                    nn.BatchNorm1d(pool_dim),
                )
            )
            in_dim = pool_dim

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.Tanh(),
        )
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for layer, res in zip(self.layers, self.residuals):
            out = layer(x)
            # Add residual
            res_out = res(x)
            x = out + res_out
        return torch.sigmoid(self.head(x))


class QCNNLightning(pl.LightningModule):
    """Lightning wrapper for QCNNModel with configurable optimizer."""

    def __init__(self, model: QCNNModel, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def QCNN() -> QCNNModel:
    """Factory returning a configured QCNNModel."""
    return QCNNModel()


__all__ = ["QCNNModel", "QCNNLightning", "QCNN"]
