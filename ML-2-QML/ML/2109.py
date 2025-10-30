import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from typing import Sequence, Union

class KernalAnsatz(nn.Module):
    """Classical RBF kernel with optional hyper‑parameter search.

    Parameters
    ----------
    gamma : float | Sequence[float], default: 1.0
        If a scalar it is used directly; if a sequence it is treated as
        a grid of candidate values for an internal ``select_best_gamma``
        routine.
    """
    def __init__(self, gamma: Union[float, Sequence[float]] = 1.0) -> None:
        super().__init__()
        if isinstance(gamma, (list, tuple, np.ndarray)):
            self.gamma_candidates = torch.tensor(gamma, dtype=torch.float32)
            self.gamma = None
        else:
            self.gamma_candidates = None
            self.gamma = float(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel between ``x`` and ``y``."""
        diff = x - y
        val = torch.sum(diff * diff, dim=-1, keepdim=True)
        return torch.exp(-self.gamma * val)

    def select_best_gamma(self, X: torch.Tensor, y: torch.Tensor,
                          cv: int = 5, reg_lambda: float = 1e-3) -> None:
        """Simple grid‑search over ``gamma_candidates`` using kernel ridge
        regression.  The chosen gamma is stored in ``self.gamma``."""
        best_score = float('inf')
        best_g = None
        for g in self.gamma_candidates:
            K = self._kernel_matrix(X, X, g)
            alpha = torch.linalg.solve(K + reg_lambda * torch.eye(K.shape[0]), y)
            preds = K @ alpha
            score = torch.mean((preds - y) ** 2).item()
            if score < best_score:
                best_score = score
                best_g = g
        self.gamma = float(best_g)

    @staticmethod
    def _kernel_matrix(X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
        diff = X.unsqueeze(1) - Y.unsqueeze(0)
        sq_norm = torch.sum(diff * diff, dim=-1)
        return torch.exp(-gamma * sq_norm)

class Kernel(nn.Module):
    """Wrapper that holds a :class:`KernalAnsatz` and exposes the same
    ``forward`` signature as the original implementation."""
    def __init__(self, gamma: Union[float, Sequence[float]] = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

    def select_best_gamma(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> None:
        """Delegate the search to the underlying ansatz."""
        self.ansatz.select_best_gamma(X, y, **kwargs)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: Union[float, Sequence[float]] = 1.0) -> np.ndarray:
    """Public helper that mirrors the original API but uses the
    flexible :class:`Kernel` implementation."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def kernel_ridge_regression(K: torch.Tensor, y: torch.Tensor,
                            reg_lambda: float = 1e-3) -> torch.Tensor:
    """Solve the kernel ridge regression closed‑form solution."""
    I = torch.eye(K.shape[0], device=K.device)
    return torch.linalg.solve(K + reg_lambda * I, y)

def kernel_regression_experiment(
        dataset_loader=load_diabetes,
        use_quantum: bool = False,
        gamma: Union[float, Sequence[float]] = 1.0,
        reg_lambda: float = 1e-3,
        test_size: float = 0.2,
        random_state: int = 42) -> dict:
    """Run a small experiment comparing the classical RBF kernel and
    the quantum variant (if requested).  Returns a dictionary with
    training and test RMSE for each kernel."""
    data = dataset_loader()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    results = {}

    # Classical RBF kernel
    if not use_quantum:
        kernel = Kernel(gamma)
        if isinstance(gamma, (list, tuple, np.ndarray)):
            kernel.select_best_gamma(X_train, y_train)
        K_train = kernel._kernel_matrix(X_train, X_train, kernel.ansatz.gamma)
        alpha = kernel_ridge_regression(K_train, y_train, reg_lambda)
        K_test = kernel._kernel_matrix(X_test, X_train, kernel.ansatz.gamma)
        preds = (K_test @ alpha).squeeze()
        rmse = mean_squared_error(y_test.squeeze().numpy(), preds.detach().numpy(), squared=False)
        results['classical'] = {'rmse': rmse, 'gamma': kernel.ansatz.gamma}
    else:
        # Quantum kernel
        try:
            from.QuantumKernelMethod_qml import Kernel as QuantumKernel
        except Exception:
            raise RuntimeError("Quantum module not available. Install torchquantum to use the quantum kernel.")
        qkernel = QuantumKernel()
        K_train = torch.tensor(kernel_matrix([x for x in X_train], [x for x in X_train]), dtype=torch.float32)
        alpha = kernel_ridge_regression(K_train, y_train, reg_lambda)
        K_test = torch.tensor(kernel_matrix([x for x in X_test], [x for x in X_train]), dtype=torch.float32)
        preds = (K_test @ alpha).squeeze()
        rmse = mean_squared_error(y_test.squeeze().numpy(), preds.detach().numpy(), squared=False)
        results['quantum'] = {'rmse': rmse}

    return results

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "kernel_ridge_regression",
    "kernel_regression_experiment",
]
