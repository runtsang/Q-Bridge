import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

class KernalAnsatz(tq.QuantumModule):
    """Parameterized quantum ansatz for kernel computation.

    Parameters
    ----------
    depth : int, default: 2
        Number of repeated layers of single‑qubit rotations and
        entangling CX gates.  Each layer uses a distinct rotation
        angle for every qubit to encode classical data.
    """
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.depth = depth

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        """Encode ``x`` and ``y`` using the circuit and return the
        overlap amplitude ``|⟨0|U(x)†U(y)|0⟩|``."""
        q_device.reset_states(x.shape[0])
        # Encode x
        for _ in range(self.depth):
            for idx in range(x.shape[1]):
                func_name_dict["ry"](q_device, wires=[idx], params=x[:, idx])
            for idx in range(x.shape[1] - 1):
                func_name_dict["cx"](q_device, wires=[idx, idx + 1])
        # Encode y in reverse
        for _ in range(self.depth):
            for idx in range(y.shape[1]):
                func_name_dict["ry"](q_device, wires=[idx], params=-y[:, idx])
            for idx in range(y.shape[1] - 1):
                func_name_dict["cx"](q_device, wires=[idx, idx + 1])

class Kernel(tq.QuantumModule):
    """Quantum kernel wrapper that exposes the same API as the
    classical counterpart."""
    def __init__(self, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  depth: int = 2) -> np.ndarray:
    """Return the Gram matrix for the quantum kernel."""
    kernel = Kernel(depth)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

def quantum_kernel_regression_experiment(
        dataset_loader=load_diabetes,
        depth: int = 2,
        reg_lambda: float = 1e-3,
        test_size: float = 0.2,
        random_state: int = 42) -> dict:
    """Run a small experiment demonstrating quantum kernel ridge
    regression on the Diabetes dataset."""
    data = dataset_loader()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    K_train = torch.tensor(kernel_matrix([x for x in X_train], [x for x in X_train], depth),
                           dtype=torch.float32)
    alpha = torch.linalg.solve(K_train + reg_lambda * torch.eye(K_train.shape[0]),
                               y_train)
    K_test = torch.tensor(kernel_matrix([x for x in X_test], [x for x in X_train], depth),
                          dtype=torch.float32)
    preds = (K_test @ alpha).squeeze()
    rmse = mean_squared_error(y_test.squeeze().numpy(), preds.detach().numpy(), squared=False)

    return {"quantum_rmse": rmse, "depth": depth}

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "quantum_kernel_regression_experiment",
]
