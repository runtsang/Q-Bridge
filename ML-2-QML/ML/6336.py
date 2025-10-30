"""Hybrid classical classifier with optional quantum‑kernel head.

The module defines a small neural network classifier and a wrapper class that
can optionally replace the linear head with a quantum kernel evaluation.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class QuantumClassifierModel(nn.Module):
    """Feed‑forward network with a quantum‑kernel head.

    Parameters
    ----------
    num_features : int
        Number of input features.
    hidden_dim : int, default 64
        Width of hidden layers.
    depth : int, default 2
        Number of hidden layers.
    use_quantum_head : bool, default False
        If True, replace the final linear layer with a quantum kernel
        estimator that acts as a support‑vector machine.
    """

    def __init__(self,
                 num_features: int,
                 hidden_dim: int = 64,
                 depth: int = 2,
                 use_quantum_head: bool = False) -> None:
        super().__init__()
        layers = [nn.Linear(num_features, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.feature_extractor = nn.Sequential(*layers)
        self.use_quantum_head = use_quantum_head
        if not use_quantum_head:
            self.classifier = nn.Linear(hidden_dim, 2)
        else:
            # placeholder for quantum kernel; actual kernel is computed in fit
            self.classifier = None
            self.scaler = StandardScaler()
            self.svm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for non‑quantum head."""
        h = self.feature_extractor(x)
        if self.use_quantum_head:
            raise RuntimeError(
                "Quantum head requires the `fit` method to build the kernel matrix.")
        return self.classifier(h)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.

        If ``use_quantum_head`` is True, a quantum kernel (RBF implemented in
        :mod:`qml_code`) is evaluated on the training data and a linear SVM
        is trained on the resulting Gram matrix. Otherwise a standard
        cross‑entropy loss is used.
        """
        if self.use_quantum_head:
            # Standardise features for stability of kernel
            X_std = self.scaler.fit_transform(X)
            # Import quantum kernel lazily to avoid heavy dependencies during
            # pure classical training.
            from qml_code import kernel_matrix  # type: ignore
            K = kernel_matrix([torch.tensor(x, dtype=torch.float32) for x in X_std],
                              [torch.tensor(x, dtype=torch.float32) for x in X_std])
            self.svm = SVC(kernel="precomputed", decision_function_shape="ovo")
            self.svm.fit(K, y)
        else:
            # Convert to torch tensors
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            self.train()
            for _ in range(200):
                optimizer.zero_grad()
                logits = self.forward(X_t)
                loss = criterion(logits, y_t)
                loss.backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.use_quantum_head:
            X_std = self.scaler.transform(X)
            K = kernel_matrix([torch.tensor(x, dtype=torch.float32) for x in X_std],
                              [torch.tensor(x, dtype=torch.float32) for x in X_std])
            return self.svm.predict(K)
        else:
            self.eval()
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32)
                logits = self.forward(X_t)
                return logits.argmax(dim=1).numpy()

def build_classifier_circuit(num_features: int,
                             depth: int,
                             use_quantum_head: bool = False) -> Tuple[nn.Module,
                                                                     Iterable[int],
                                                                     Iterable[int],
                                                                     list[int]]:
    """Return a classifier instance and metadata.

    The signature mirrors the original ``build_classifier_circuit`` so that
    downstream code can import it unchanged.  ``encoding`` and ``observables``
    are placeholders kept for API compatibility; they are not used by the
    returned network.
    """
    net = QuantumClassifierModel(num_features,
                                 hidden_dim=num_features,
                                 depth=depth,
                                 use_quantum_head=use_quantum_head)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in net.parameters()]
    observables = [0, 1]
    return net, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
