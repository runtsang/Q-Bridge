"""QuantumKernel: quantum kernel implementation with entanglement and flexible ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, List, Dict, Optional

class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that evaluates the fidelity between two states prepared from
    classical inputs. The ansatz is fully programmable and includes entangling
    gates to increase the expressive power of the feature map.
    """

    def __init__(
        self,
        n_wires: int = 4,
        ansatz: Optional[List[Dict]] = None,
        measurement: str = "amplitude",
        shots: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        n_wires : int
            Number of qubits in the device.
        ansatz : list[dict], optional
            List of gate specifications. Each dict must contain:
                - ``func``: name of the TorchQuantum gate (e.g., 'ry', 'cx').
                - ``wires``: list[int] of target wires.
                - ``input_idx``: list[int] indices into the input vector for parameterised gates.
            If None, a default RY‑only ansatz with an entangling layer is used.
        measurement : {"amplitude", "probability"}
            Measurement type. ``amplitude`` returns the magnitude of the |0⟩ state.
        shots : int
            Number of shots for probability measurements; ignored for amplitude.
        """
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.measurement = measurement
        self.shots = shots

        if ansatz is None:
            ansatz = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
                # Entangling layer
                {"input_idx": [], "func": "cx", "wires": [0, 1]},
                {"input_idx": [], "func": "cx", "wires": [1, 2]},
                {"input_idx": [], "func": "cx", "wires": [2, 3]},
            ]
        self.ansatz = ansatz

    @tq.static_support
    def _apply_ansatz(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode x and uncompute y on the same device.
        """
        # Encode x
        for info in self.ansatz:
            params = (
                x[:, info["input_idx"]]
                if info["input_idx"]
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

        # Uncompute with y (reverse order, negative params)
        for info in reversed(self.ansatz):
            params = (
                -y[:, info["input_idx"]]
                if info["input_idx"]
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Public API that returns the kernel value for two 1‑D tensors.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(x.shape[0])
        self._apply_ansatz(self.q_device, x, y)

        if self.measurement == "amplitude":
            return torch.abs(self.q_device.states.view(-1)[0])
        if self.measurement == "probability":
            probs = torch.abs(self.q_device.states.view(-1)) ** 2
            return probs[0]
        raise ValueError(f"Unsupported measurement: {self.measurement}")

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of feature vectors.
        """
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def __repr__(self) -> str:
        return f"<QuantumKernel wires={self.n_wires} measurement={self.measurement}>"

__all__ = ["QuantumKernel"]
