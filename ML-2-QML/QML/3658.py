import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class RegressionDataset(Dataset):
    """Quantum‑style dataset that samples superposition states."""
    def __init__(self, samples: int, num_wires: int = 4):
        self.states, self.labels = self._generate_states(samples, num_wires)

    def _generate_states(self, samples: int, num_wires: int):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states.astype(np.complex64), labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class QuantumKernel(tq.QuantumModule):
    """Variational quantum kernel using a simple rotation ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.ry_ops = nn.ModuleList([tq.RY(has_params=True, trainable=False) for _ in range(n_wires)])

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x, then de‑encode y in reverse order."""
        qdev.reset_states(x.shape[0])
        for i in range(self.n_wires):
            self.ry_ops[i](qdev, wires=[i], params=x[:, i])
        for i in reversed(range(self.n_wires)):
            self.ry_ops[i](qdev, wires=[i], params=-y[:, i])

class HybridKernelRegression(tq.QuantumModule):
    """
    Quantum kernel regression.
    Computes a kernel matrix via a variational quantum circuit and fits a Ridge model.
    """
    def __init__(self, n_wires: int = 4, ridge_alpha: float = 1.0):
        super().__init__()
        self.n_wires = n_wires
        self.kernel = QuantumKernel(n_wires)
        self.ridge_alpha = ridge_alpha
        self._ridge = None

    def _quantum_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix by looping over all pairs."""
        n_x, n_y = X.shape[0], Y.shape[0]
        K = torch.zeros(n_x, n_y, device=X.device)
        for i in range(n_x):
            for j in range(n_y):
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=X.device)
                self.kernel(qdev, X[i:i+1], Y[j:j+1])
                K[i, j] = torch.abs(qdev.states[0, 0])
        return K

    def fit(self, dataset: Dataset) -> None:
        """Fit a Ridge regression on the quantum kernel matrix."""
        X = torch.stack([item["states"] for item in dataset], dim=0).float()
        y = torch.stack([item["target"] for item in dataset], dim=0)
        K = self._quantum_kernel_matrix(X, X).cpu().numpy()
        self._ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        self._ridge.fit(K, y.cpu().numpy())

    def predict(self, dataset: Dataset) -> torch.Tensor:
        """Predict using the fitted Ridge regression on the quantum kernel matrix."""
        if self._ridge is None:
            raise RuntimeError("Model must be trained before calling predict.")
        X = torch.stack([item["states"] for item in dataset], dim=0).float()
        K = self._quantum_kernel_matrix(X, X)
        y_pred = self._ridge.predict(K.cpu().numpy())
        return torch.tensor(y_pred, device=X.device)

    def evaluate(self, dataset: Dataset) -> float:
        """Return RMSE on the given dataset."""
        y_true = torch.stack([item["target"] for item in dataset], dim=0)
        y_pred = self.predict(dataset)
        rmse = torch.sqrt(mean_squared_error(y_true.cpu().numpy(), y_pred.cpu().numpy()))
        return float(rmse)

__all__ = ["HybridKernelRegression", "RegressionDataset"]
