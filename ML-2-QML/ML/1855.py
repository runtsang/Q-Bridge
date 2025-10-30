"""Hybrid quantum‑classical kernel module with trainable variational ansatz."""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Optional

__all__ = ["QuantumKernelMethod", "QuantumKernelMethodConfig"]

class QuantumKernelMethodConfig:
    """Configuration dataclass for the hybrid kernel model."""
    def __init__(
        self,
        gamma: float = 1.0,
        n_wires: int = 4,
        lr: float = 0.01,
        epochs: int = 200,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.n_wires = n_wires
        self.lr = lr
        self.epochs = epochs
        self.device = device

class KernalAnsatz(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuantumAnsatz(nn.Module):
    """Trainable variational ansatz using TorchQuantum."""
    def __init__(self, n_wires: int):
        super().__init__()
        import torchquantum as tq
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        # One rotation parameter per wire
        self.params = nn.Parameter(torch.randn(n_wires))
        self.wires = list(range(n_wires))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode data and return the device state."""
        import torchquantum as tq
        self.q_device.reset_states(x.shape[0])
        for wire, param in zip(self.wires, self.params):
            tq.ry(self.q_device, wires=wire, params=x[:, wire] + param)
        return self.q_device.states

class QuantumKernelMethod(nn.Module):
    """Hybrid kernel that can operate in classical or quantum mode."""
    def __init__(self, config: QuantumKernelMethodConfig, mode: str = "classic"):
        super().__init__()
        self.mode = mode
        self.config = config
        self.device = torch.device(config.device)
        if mode == "classic":
            self.kernel = KernalAnsatz(gamma=config.gamma).to(self.device)
        elif mode == "quantum":
            self.kernel = QuantumAnsatz(n_wires=config.n_wires).to(self.device)
        else:
            raise ValueError("mode must be 'classic' or 'quantum'")
        self.svm: Optional[SVC] = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value between two samples."""
        x = x.to(self.device)
        y = y.to(self.device)
        if self.mode == "classic":
            return self.kernel(x, y).squeeze()
        else:  # quantum
            x_state = self.kernel(x)
            y_state = self.kernel(y)
            # absolute overlap
            overlap = torch.abs(torch.sum(x_state.conj() * y_state, dim=-1))
            return overlap

    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between two datasets."""
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        n, m = X_t.shape[0], Y_t.shape[0]
        K = torch.empty((n, m), device=self.device)
        for i in range(n):
            for j in range(m):
                K[i, j] = self.forward(X_t[i].unsqueeze(0), Y_t[j].unsqueeze(0))
        return K.cpu().numpy()

    def train_kernel(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the quantum ansatz with a contrastive loss."""
        if self.mode!= "quantum":
            return
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        optimizer = Adam(self.kernel.parameters(), lr=self.config.lr)
        for epoch in range(self.config.epochs):
            loss = 0.0
            for i in range(len(X_t)):
                xi = X_t[i].unsqueeze(0)
                for j in range(i + 1, len(X_t)):
                    xj = X_t[j].unsqueeze(0)
                    label = (y_t[i] == y_t[j]).float()
                    k = self.forward(xi, xj)
                    loss += label * (1 - k) ** 2 + (1 - label) * k ** 2
            loss = loss / (len(X_t) * (len(X_t) - 1) / 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 50 == 0:
                print(f"[QuantumKernel] epoch {epoch} loss {loss.item():.4f}")

    def fit(self, X: np.ndarray, y: np.ndarray, C: float = 1.0) -> None:
        """Fit an SVM using the pre‑computed kernel matrix."""
        K = self.kernel_matrix(X, X)
        self.svm = SVC(C=C, kernel="precomputed")
        self.svm.fit(K, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for new samples."""
        if self.svm is None:
            raise RuntimeError("Model has not been fitted.")
        K = self.kernel_matrix(X, self.svm.support_vectors_)
        return self.svm.predict(K)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
