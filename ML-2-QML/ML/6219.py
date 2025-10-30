import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from typing import Optional

class EstimatorQNN(nn.Module):
    """
    A robust regression model that incorporates feature scaling, dropout,
    L2 regularisation, and an optional early‑stopping mechanism.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: list[int] | None = None,
        output_dim: int = 1,
        dropout: float = 0.1,
        l2_reg: float = 1e-3,
    ) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.l2_reg = l2_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect pre‑scaled input; scaling is applied in fit
        return self.network(x)

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        val_split: float = 0.2,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        patience: int = 20,
        device: str = "cpu",
    ) -> None:
        """
        Train the model with early‑stopping and L2 regularisation.
        """
        X, y = X.to(device), y.to(device)
        # Fit scaler on training data
        X_np = X.cpu().numpy()
        self.scaler.fit(X_np)
        X_scaled = torch.tensor(self.scaler.transform(X_np), dtype=torch.float32, device=device)

        # Create validation split
        n = X_scaled.shape[0]
        idx = torch.randperm(n)
        val_idx = idx[: int(val_split * n)]
        train_idx = idx[int(val_split * n) :]

        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_val, y_val = X_scaled[val_idx], y[val_idx]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.l2_reg)
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state: Optional[dict] = None
        counter = 0

        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                batch_idx = perm[i : i + batch_size]
                xb, yb = X_train[batch_idx], y_train[batch_idx]
                optimizer.zero_grad()
                preds = self.forward(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.eval()
            with torch.no_grad():
                val_pred = self.forward(X_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        if best_state:
            self.load_state_dict(best_state)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return predictions for scaled inputs.
        """
        X_np = X.cpu().numpy()
        X_scaled = torch.tensor(self.scaler.transform(X_np), dtype=torch.float32, device=X.device)
        self.eval()
        with torch.no_grad():
            return self.forward(X_scaled)
