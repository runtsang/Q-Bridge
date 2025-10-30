import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Iterable

class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QFCModel(nn.Module):
    """Classical CNN followed by a fully connected projection to four features."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

class FCL(nn.Module):
    """Fully connected layer that can be used as a quantum‑style parameterized layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class SelfAttention(nn.Module):
    """Classical self‑attention block mimicking the quantum interface."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class HybridKernelModel(nn.Module):
    """
    Hybrid kernel model that can operate in either classical or quantum mode.
    In classical mode it uses an RBF kernel; in quantum mode it uses a
    TorchQuantum kernel with optional feature extraction via QFCModel.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        use_quantum: bool = False,
        n_features: int = 4,
        attention: bool = False,
        attention_params: dict | None = None,
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.gamma = gamma
        self.n_features = n_features
        self.attention = attention
        self.attention_params = attention_params or {}

        # Classical components
        self.kernel = Kernel(gamma)
        self.qfc = QFCModel()
        self.fcl = FCL(n_features)
        self.self_attention = SelfAttention(embed_dim=4) if attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid kernel matrix between input x and the internal
        training data set. For demonstration purposes the training set is
        assumed to be the same as x, yielding a Gram matrix.
        """
        # Feature extraction
        features = self.qfc(x)
        if self.attention:
            # Dummy rotation and entangle params
            rot = np.random.randn(self.n_features, self.n_features)
            ent = np.random.randn(self.n_features - 1)
            features_np = self.self_attention.run(rot, ent, features.detach().cpu().numpy())
            features = torch.from_numpy(features_np).float()

        # Apply fully connected layer
        thetas = np.linspace(0, np.pi, self.n_features)
        _ = self.fcl.run(thetas)  # expectation value is not used in this simple demo

        # Compute kernel matrix
        if self.use_quantum:
            # Placeholder: quantum kernel would be implemented in qml_code
            return self.kernel(features, features)
        else:
            return self.kernel(features, features)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        if self.use_quantum:
            # Placeholder: quantum kernel would be implemented in qml_code
            return kernel_matrix(a, b, self.gamma)
        else:
            return kernel_matrix(a, b, self.gamma)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QFCModel",
    "FCL",
    "SelfAttention",
    "HybridKernelModel",
]
