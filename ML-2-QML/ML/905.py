import torch
from torch import nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """
    A flexible, regularized QCNN‑inspired classifier.
    Supports dynamic depth, optional dropout and batch‑norm,
    and a built‑in training helper that works with any DataLoader.
    """
    def __init__(self, input_dim: int = 8, hidden_dims: list[int] | None = None,
                 dropout: float = 0.0, use_batch_norm: bool = False) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 12, 8, 4, 4]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).round()

    def train_loop(self, train_loader, val_loader,
                   epochs: int = 20, lr: float = 1e-3,
                   weight_decay: float = 0.0,
                   patience: int = 5,
                   device: str | torch.device = "cpu") -> dict[str, list[float]]:
        """
        Simple training loop with early stopping.
        Returns history dict with 'train_loss', 'val_loss', 'val_acc'.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val = float("inf")
        counter = 0
        for epoch in range(epochs):
            self.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                optimizer.zero_grad()
                out = self(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = sum(train_losses) / len(train_losses)

            self.eval()
            val_losses, correct, total = [], 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device).float()
                    out = self(xb).squeeze()
                    loss = criterion(out, yb)
                    val_losses.append(loss.item())
                    preds = out.round()
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_loss = sum(val_losses) / len(val_losses)
            val_acc = correct / total

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                torch.save(self.state_dict(), "best_qcnn_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    break
        # load best
        self.load_state_dict(torch.load("best_qcnn_model.pt"))
        return history

def QCNN(**kwargs):
    return QCNNModel(**kwargs)

__all__ = ["QCNNModel", "QCNN"]
