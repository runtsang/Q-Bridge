"""QuantumKernelMethod: quantum kernel with flexible rotation‑entanglement ansatz.

The quantum implementation builds a parameterised ansatz that can be
configured for depth, wire count and entanglement strategy.  It can run
on CPU or GPU back‑ends and exposes the same public API as the classical
counterpart.
"""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, List, Dict

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel evaluated via a configurable rotation‑entanglement ansatz.

    Parameters
    ----------
    n_wires : int, default=4
        Number of qubits in the device.
    depth : int, default=2
        Number of rotation‑entanglement layers.
    entanglement : str, default="full"
        Entanglement pattern: "full" or "nearest".
    device_type : str, default="cpu"
        Backend device: "cpu" or "cuda".
    """

    def __init__(
        self,
        n_wires: int = 4,
        depth: int = 2,
        entanglement: str = "full",
        device_type: str = "cpu",
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.entanglement = entanglement
        self.q_device = tq.QuantumDevice(
            n_wires=self.n_wires, device_type=device_type
        )
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self) -> List[Dict]:
        """Construct a list of gate specifications."""
        ops: List[Dict] = []
        for _ in range(self.depth):
            # Rotation layer
            for w in range(self.n_wires):
                ops.append(
                    {"input_idx": [w], "func": "ry", "wires": [w]}
                )
            # Entanglement layer
            if self.entanglement == "full":
                for w in range(self.n_wires):
                    ops.append(
                        {"input_idx": [], "func": "cx", "wires": [w, (w + 1) % self.n_wires]}
                    )
            else:  # nearest‑neighbour
                for w in range(self.n_wires - 1):
                    ops.append(
                        {"input_idx": [], "func": "cx", "wires": [w, w + 1]}
                    )
        return ops

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Encode two classical vectors and return the magnitude of the first
        amplitude of the resulting state vector.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])

        # Encode x
        for info in self.ansatz:
            params = x[:, info["input_idx"]] if info["input_idx"] else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Encode -y (reverse direction)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]] if info["input_idx"] else None
            func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)

        # Return overlap magnitude
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two sets of classical vectors."""
        mat = torch.empty((len(a), len(b)), dtype=torch.float32)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = self.forward(xi, yj)
        return mat.cpu().numpy()


__all__ = ["QuantumKernelMethod"]
