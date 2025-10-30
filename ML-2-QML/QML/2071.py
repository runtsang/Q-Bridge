import pennylane as qml
import torch
import numpy as np
from typing import Sequence, Optional

class QuantumKernelMethod:
    """
    Quantum kernel based on a variational feature map implemented with PennyLane.
    The kernel is the squared fidelity |<ψ(x)|ψ(y)>|^2 of two quantum states prepared by a
    trainable ansatz.  The ansatz parameters can be optimized with a simple gradient‑descent
    routine, turning the kernel into a learnable feature map.
    The module is fully Torch‑compatible and can be used in hybrid pipelines.
    """
    def __init__(self, n_wires: int = 4, device: Optional[qml.Device] = None):
        self.n_wires = n_wires
        self.device = device or qml.device("default.qubit", wires=self.n_wires)
        # Trainable parameters: 2 layers of rotations + CNOTs
        self.params = torch.randn((2, self.n_wires, 3), requires_grad=True, dtype=torch.float32)
        self._build_qnode()

    def _ansatz(self, x):
        # Data encoding: RX on each qubit
        for w in range(self.n_wires):
            qml.RX(x[w], wires=w)
        # Parameterized layers
        for layer in range(self.params.shape[0]):
            for w in range(self.n_wires):
                qml.Rot(*self.params[layer, w], wires=w)
            # Entangling CNOTs in a chain
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])

    def _state_qnode(self):
        @qml.qnode(self.device, interface="torch")
        def _state(x: torch.Tensor):
            self._ansatz(x)
            return qml.state()
        return _state

    def _build_qnode(self):
        self.state_qnode = self._state_qnode()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel k(x, y) = |<ψ(x)|ψ(y)>|^2.
        """
        state_x = self.state_qnode(x)
        state_y = self.state_qnode(y)
        return torch.abs(torch.dot(state_x, torch.conj(state_y))) ** 2

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix for two collections of data points.
        """
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def fit(self, a: Sequence[torch.Tensor], labels: Sequence[int], lr: float = 0.01, epochs: int = 100):
        """
        Train the ansatz parameters to encourage high similarity for same‑class data
        and low similarity for different‑class data.
        """
        optimizer = torch.optim.Adam([self.params], lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = 0.0
            for i, xi in enumerate(a):
                for j, xj in enumerate(a):
                    k = self.forward(xi, xj)
                    if labels[i] == labels[j]:
                        loss -= torch.log(k + 1e-8)
                    else:
                        loss += torch.log(k + 1e-8)
            loss.backward()
            optimizer.step()

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.forward(x, y)

__all__ = ["QuantumKernelMethod"]
