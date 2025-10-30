import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class RegressionDataset(Dataset):
    """Dataset that mimics the quantum superposition data but stays purely classical."""
    def __init__(self, samples: int, num_features: int = 4):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        self.features = torch.tensor(x, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {"states": self.features[idx], "target": self.labels[idx]}

class HybridKernelRegression(nn.Module):
    """
    Classical RBF kernel + Ridge regression.
    The kernel matrix is computed on the fly; no quantum part is involved.
    """
    def __init__(self, gamma: float = 1.0, ridge_alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.ridge_alpha = ridge_alpha
        self._ridge = None  # will be instantiated after fitting

    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel matrix in a numerically stable way."""
        X2 = (X ** 2).sum(dim=1, keepdim=True)
        Y2 = (Y ** 2).sum(dim=1, keepdim=True).t()
        K = torch.exp(-self.gamma * (X2 + Y2 - 2 * X @ Y.t()))
        return K

    def fit(self, dataset: Dataset) -> None:
        """Fit a Ridge regression model on the kernel matrix."""
        X = torch.stack([item["states"] for item in dataset], dim=0)
        y = torch.stack([item["target"] for item in dataset], dim=0)
        K = self._rbf_kernel(X, X).cpu().numpy()
        self._ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        self._ridge.fit(K, y.cpu().numpy())

    def predict(self, dataset: Dataset) -> torch.Tensor:
        """Predict using the fitted Ridge regression on the kernel matrix."""
        if self._ridge is None:
            raise RuntimeError("Model must be trained before calling predict.")
        X = torch.stack([item["states"] for item in dataset], dim=0)
        K = self._rbf_kernel(X, X)
        y_pred = self._ridge.predict(K.cpu().numpy())
        return torch.tensor(y_pred, device=X.device)

    def evaluate(self, dataset: Dataset) -> float:
        """Return RMSE on the given dataset."""
        y_true = torch.stack([item["target"] for item in dataset], dim=0)
        y_pred = self.predict(dataset)
        rmse = torch.sqrt(mean_squared_error(y_true.cpu().numpy(), y_pred.cpu().numpy()))
        return float(rmse)

__all__ = ["HybridKernelRegression", "RegressionDataset"]
