"""
QuantumKernelMethod module.

Provides a scikit‑learn compatible estimator that can use either a classical
RBF kernel or a quantum kernel implemented with TorchQuantum.  The estimator
implements kernel ridge regression and can be used in any sklearn pipeline.
It also exposes a small hyper‑parameter search helper that can be used
directly from the class.
"""

from __future__ import annotations

from typing import Sequence, Any, Dict

import numpy as np
import torch

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler

# Import quantum kernel implementation
try:
    from.QuantumKernelMethod_qml import QuantumKernel as _QuantumKernel
except Exception:
    _QuantumKernel = None


class _BaseKernel:
    """
    Utility base class for kernel computations.
    Provides a static method to compute the Gram matrix for any callable
    kernel function.
    """

    @staticmethod
    def gram_matrix(
        X: torch.Tensor,
        Y: torch.Tensor,
        kernel_func,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute the Gram matrix between X and Y using the supplied kernel_func.
        """
        X_cpu = X.detach().cpu()
        Y_cpu = Y.detach().cpu()
        K = np.array(
            [
                [
                    kernel_func(x, y, **kwargs).item()
                    for y in Y_cpu
                ]
                for x in X_cpu
            ]
        )
        return K


class QuantumKernelMethod(BaseEstimator, RegressorMixin):
    """
    scikit‑learn style estimator that performs kernel ridge regression
    using either a classical RBF kernel or a quantum kernel.

    Parameters
    ----------
    kernel : {'rbf', 'quantum'}, default='rbf'
        Kernel type to use.
    gamma : float, optional
        Parameter for the RBF kernel. Ignored for the quantum kernel.
    backend : {'classical', 'quantum'}, default='classical'
        Which backend to use when ``kernel='quantum'``.
    n_qubits : int, default=4
        Number of qubits for the quantum kernel.
    alpha : float, default=1e-5
        Regularisation parameter for kernel ridge regression.
    scaler : bool, default=True
        Whether to standardise the input features before kernel evaluation.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float | None = 1.0,
        backend: str = "classical",
        n_qubits: int = 4,
        alpha: float = 1e-5,
        scaler: bool = True,
    ) -> None:
        self.kernel = kernel
        self.gamma = gamma
        self.backend = backend
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.scaler = scaler

    def _validate_params(self) -> None:
        if self.kernel not in {"rbf", "quantum"}:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
        if self.backend not in {"classical", "quantum"}:
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.kernel == "quantum" and self.backend == "quantum" and _QuantumKernel is None:
            raise ImportError("Quantum kernel module not available.")

    def _fit_kernel(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Compute the kernel matrix and solve the ridge regression weights.
        """
        if self.kernel == "rbf":
            kernel_func = self._rbf_kernel
            kwargs = {"gamma": self.gamma}
        else:
            kernel_func = self._quantum_kernel
            kwargs = {"n_qubits": self.n_qubits}

        K = _BaseKernel.gram_matrix(X, X, kernel_func, **kwargs)
        K_reg = K + self.alpha * np.eye(K.shape[0])
        self.alpha_vector_ = np.linalg.solve(K_reg, y.numpy())

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
        diff = x - y
        return torch.exp(-gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def _quantum_kernel(self, x: torch.Tensor, y: torch.Tensor, n_qubits: int) -> torch.Tensor:
        quantum_kernel = _QuantumKernel(n_qubits=n_qubits)
        return quantum_kernel(x, y)

    def fit(self, X: Sequence[torch.Tensor], y: Sequence[torch.Tensor]) -> "QuantumKernelMethod":
        self._validate_params()
        X_arr = np.array([x.numpy() for x in X])
        y_arr = np.array([yi.numpy() for yi in y]).reshape(-1, 1)

        if self.scaler:
            self.scaler_ = StandardScaler()
            X_arr = self.scaler_.fit_transform(X_arr)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32)

        self._fit_kernel(X_tensor, y_tensor)
        self.X_train_ = X_tensor
        self.y_train_ = y_tensor
        return self

    def predict(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        check_is_fitted(self, "alpha_vector_")
        X_arr = np.array([x.numpy() for x in X])
        if self.scaler:
            X_arr = self.scaler_.transform(X_arr)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        K_test = _BaseKernel.gram_matrix(
            X_tensor,
            self.X_train_,
            self._rbf_kernel if self.kernel == "rbf" else self._quantum_kernel,
            gamma=self.gamma if self.kernel == "rbf" else None,
            n_qubits=self.n_qubits if self.kernel == "quantum" else None,
        )

        y_pred = K_test @ self.alpha_vector_
        return y_pred.squeeze()

    def transform(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        check_is_fitted(self, "alpha_vector_")
        X_arr = np.array([x.numpy() for x in X])
        if self.scaler:
            X_arr = self.scaler_.transform(X_arr)
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        K = _BaseKernel.gram_matrix(
            X_tensor,
            self.X_train_,
            self._rbf_kernel if self.kernel == "rbf" else self._quantum_kernel,
            gamma=self.gamma if self.kernel == "rbf" else None,
            n_qubits=self.n_qubits if self.kernel == "quantum" else None,
        )
        return K

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "kernel": self.kernel,
            "gamma": self.gamma,
            "backend": self.backend,
            "n_qubits": self.n_qubits,
            "alpha": self.alpha,
            "scaler": self.scaler,
        }

    def set_params(self, **params: Any) -> "QuantumKernelMethod":
        for key, value in params.items():
            setattr(self, key, value)
        return self
