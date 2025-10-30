import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

class QuantumRegression__gen104(nn.Module):
    """
    Classical regression model that supports:
      * Feature selection via SelectKBest
      * Early‑stopping on a validation set
      * Training with a DataLoader
    """
    def __init__(self, num_features: int, hidden_sizes=[32, 16], k_best: int = 10,
                 patience: int = 5, lr: float = 1e-3, device: torch.device | str | None = None):
        super().__init__()
        self.device = torch.device(device or "cpu")
        self.k_best = k_best
        self.patience = patience
        self.lr = lr
        self.scaler = StandardScaler()
        self.selector = None
        # placeholder network; will be rebuilt after feature selection
        self.net = nn.Sequential()
        self._build_network(num_features)

    @staticmethod
    def generate_superposition_data(num_features: int, samples: int):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    class RegressionDataset(Dataset):
        def __init__(self, samples: int, num_features: int):
            self.features, self.labels = self.generate_superposition_data(num_features, samples)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return torch.tensor(self.features[idx], dtype=torch.float32), \
                   torch.tensor(self.labels[idx], dtype=torch.float32)

    def _build_network(self, input_dim: int):
        layers = []
        in_dim = input_dim
        for h in [32, 16]:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).to(self.device)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None,
            epochs: int = 100) -> None:
        # Gather all training data to fit scaler & selector
        X_all, y_all = [], []
        for X, y in train_loader:
            X_all.append(X.cpu().numpy())
            y_all.append(y.cpu().numpy())
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        # Standardise and select features
        X_all = self.scaler.fit_transform(X_all)
        if self.k_best < X_all.shape[1]:
            self.selector = SelectKBest(score_func=f_regression, k=self.k_best)
            X_all = self.selector.fit_transform(X_all, y_all)
        else:
            self.selector = None

        # Re‑build network to match new input dimension
        self._build_network(X_all.shape[1])

        # Create new DataLoader from transformed data
        train_dataset = TensorDataset(torch.tensor(X_all, dtype=torch.float32),
                                      torch.tensor(y_all, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        best_val = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                preds = self.net(X).squeeze(-1)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X.size(0)
            epoch_loss /= len(train_loader.dataset)

            if val_loader is not None:
                val_loss = self._evaluate(val_loader, criterion)
                if val_loss < best_val:
                    best_val = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    break

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> float:
        self.net.eval()
        total = 0.0
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = self.net(X).squeeze(-1)
                loss = criterion(preds, y)
                total += loss.item() * X.size(0)
        self.net.train()
        return total / len(loader.dataset)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            X_t = X.to(self.device)
            X_t = self.scaler.transform(X_t.cpu().numpy())
            if self.selector is not None:
                X_t = self.selector.transform(X_t)
            X_t = torch.tensor(X_t, dtype=torch.float32, device=self.device)
            return self.net(X_t).squeeze(-1).cpu()

    def predict_batch(self, loader: DataLoader) -> torch.Tensor:
        self.net.eval()
        preds = []
        with torch.no_grad():
            for X, _ in loader:
                preds.append(self.predict(X))
        return torch.cat(preds, dim=0)

__all__ = ["QuantumRegression__gen104"]
