import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class RBFKernel(tq.QuantumModule):
    """Quantum kernel using a parameterized ansatz.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubits. Default 4.
    depth : int, optional
        Number of repetitions of the entangling layer. Default 2.
    entanglement : str, optional
        Entanglement pattern: 'full', 'circular', or 'linear'. Default 'full'.
    """

    def __init__(
        self,
        n_wires: int = 4,
        depth: int = 2,
        entanglement: str = "full",
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.entanglement = entanglement
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.layers = self._build_ansatz()

    def _build_ansatz(self):
        """Construct a layered ansatz with RX rotations and controlled‑Z entanglement."""
        layers = []
        for _ in range(self.depth):
            # RX rotations encoding classical data
            for w in range(self.n_wires):
                layers.append(
                    {"input_idx": [w], "func": "rx", "wires": [w]}
                )
            # Entangling layer
            if self.entanglement == "full":
                for i in range(self.n_wires):
                    for j in range(i + 1, self.n_wires):
                        layers.append(
                            {"input_idx": [], "func": "cz", "wires": [i, j]}
                        )
            elif self.entanglement == "circular":
                for i in range(self.n_wires):
                    layers.append(
                        {"input_idx": [], "func": "cz", "wires": [i, (i + 1) % self.n_wires]}
                    )
            else:  # linear
                for i in range(self.n_wires - 1):
                    layers.append(
                        {"input_idx": [], "func": "cz", "wires": [i, i + 1]}
                    )
        return layers

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Apply the ansatz with data x and -y to compute overlap."""
        # Reset device
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.layers:
            params = x[:, info["input_idx"]] if info["input_idx"] else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode -y
        for info in reversed(self.layers):
            params = -y[:, info["input_idx"]] if info["input_idx"] else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def forward_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        # Overlap squared: |<0|U(x)U†(y)|0>|^2
        overlap = torch.abs(self.q_device.states.view(-1)[0]) ** 2
        return overlap

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix between two batches."""
        return self.forward_kernel(a, b)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix between two sequences of tensors."""
    kernel = RBFKernel()
    return np.array([[kernel.forward_kernel(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["RBFKernel", "kernel_matrix"]
