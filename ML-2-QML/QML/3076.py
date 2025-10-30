"""Quantum implementation of the quanvolution filter and kernel.

The class `QuanvolutionHybrid` is a pure quantum module that implements a
fixed Ry‑based encoding with a random layer and returns the absolute
value of the inner product of two encoded states.  It can be used as a
drop‑in replacement for the classical QuanvolutionFilter.  The static
method `kernel_matrix` evaluates the Gram matrix between two batches of
4‑dimensional vectors via the same encoder, enabling kernel‑based
learning in a fully quantum setting.
"""

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum kernel based on a fixed Ry encoding with a random layer.

    The module accepts two 4‑dim vectors (x and y) and returns the absolute
    value of the inner product of the two corresponding quantum states.
    The static method ``kernel_matrix`` evaluates the Gram matrix between
    two batches of vectors.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x and y on the same device and return overlap."""
        # Encode x
        self.encoder(q_device, x)
        self.q_layer(q_device)
        # Encode y with negative parameters
        for info in reversed(self.encoder.layers):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Overlap measurement
        return torch.abs(q_device.states.view(-1)[0])

    @staticmethod
    def kernel_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix between two batches of 4‑dim vectors."""
        device = a.device
        n_wires = 4
        qdev = tq.QuantumDevice(n_wires, bsz=a.shape[0] + b.shape[0], device=device)
        kernel = QuanvolutionHybrid()
        result = torch.empty(a.shape[0], b.shape[0], device=device)
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                qdev.reset_states(1)
                result[i, j] = kernel(qdev, a[i].unsqueeze(0), b[j].unsqueeze(0))
        return result

__all__ = ["QuanvolutionHybrid"]
