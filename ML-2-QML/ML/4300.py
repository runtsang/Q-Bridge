"""Classical classifier model integrating feed‑forward network, optional RBF kernel, and a fully‑connected layer.

The class mirrors the quantum helper interface while offering richer feature engineering.  It can
be used as a stand‑alone model or as a component in a hybrid pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, List, Sequence

class FCL(nn.Module):
    """Simple fully‑connected layer that returns a scalar expectation."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # expectation = tanh(linear(thetas))
        return torch.tanh(self.linear(thetas)).view(-1)

class RBFKernel:
    """Radial‑basis‑function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QuantumClassifierModelGen:
    """Hybrid classical classifier with optional kernel and FCL."""
    def __init__(self,
                 num_features: int,
                 depth: int = 3,
                 use_kernel: bool = False,
                 gamma: float = 1.0,
                 use_fcl: bool = False,
                 fcl_dim: int = 1) -> None:
        self.num_features = num_features
        self.depth = depth
        self.use_kernel = use_kernel
        self.gamma = gamma
        self.use_fcl = use_fcl
        self.fcl_dim = fcl_dim

        self.network, self.encoding, self.weight_sizes, self.observables = self.build_classifier_circuit()

        if self.use_fcl:
            self.fcl = FCL(self.fcl_dim)
        else:
            self.fcl = None

    def build_classifier_circuit(self) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        layers: List[nn.Module] = []
        in_dim = self.num_features
        encoding = list(range(self.num_features))
        weight_sizes: List[int] = []

        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = self.num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables = list(range(2))
        return network, encoding, weight_sizes, observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_kernel:
            # compute kernel feature vector
            k = np.array([kernel_matrix([x], [x], self.gamma)[0,0]])
            x = torch.tensor(k, dtype=torch.float32).view(1, -1)
        if self.use_fcl:
            # run FCL on a random theta vector for demonstration
            theta = torch.randn(self.fcl_dim)
            fcl_out = self.fcl(theta)
            x = torch.cat([x, fcl_out], dim=-1)
        return self.network(x)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            lr: float = 1e-3,
            device: str = "cpu") -> None:
        self.network.to(device)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            probs = torch.softmax(self.forward(X_t), dim=-1)
            return probs.argmax(dim=-1).cpu().numpy()
