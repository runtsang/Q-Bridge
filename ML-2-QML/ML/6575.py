import torch
from torch import nn
import numpy as np
from.QuantumKernelMethod import Kernel as ClassicalKernel

class HybridKernelRegressor(nn.Module):
    """
    A hybrid regressor that fuses a classical RBF kernel with a quantum kernel
    and learns a mapping from the combined feature space to the target.
    """

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.classical_kernel = ClassicalKernel(gamma)
        self.quantum_kernel = None  # to be injected from QML side
        self.mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def integrate_quantum_kernel(self, quantum_kernel):
        """
        Attach a quantum kernel implementation.
        """
        self.quantum_kernel = quantum_kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the combined kernel between x and y and pass through MLP.
        """
        if self.quantum_kernel is None:
            raise RuntimeError("Quantum kernel not integrated")
        c_feat = self.classical_kernel(x, y).unsqueeze(-1)
        q_feat = self.quantum_kernel(x, y).unsqueeze(-1)
        feat = torch.cat([c_feat, q_feat], dim=-1)
        return self.mlp(feat)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Store training data and compute kernel matrices for future predictions.
        """
        self.X_train = X.detach().clone()
        self.y_train = y.detach().clone()
        self.Kc = self.classical_kernel(X, X)
        self.Kq = self.quantum_kernel(X, X)
        self.K = torch.cat([self.Kc.unsqueeze(-1), self.Kq.unsqueeze(-1)], dim=-1)
        lam = 1e-3
        K = self.K + lam * torch.eye(self.K.shape[0], device=self.K.device)
        self.alpha = torch.linalg.solve(K, self.y_train)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using stored training data and learned coefficients.
        """
        Kc = self.classical_kernel(X, self.X_train)
        Kq = self.quantum_kernel(X, self.X_train)
        K = torch.cat([Kc.unsqueeze(-1), Kq.unsqueeze(-1)], dim=-1)
        return K @ self.alpha

__all__ = ["HybridKernelRegressor"]
