"""Hybrid quantum kernel method combining a quantum convolutional filter with a
quantum kernel ansatz.

The module defines :class:`HybridKernel`, a quantum‑centric equivalent of the
classical ``HybridKernel``.  It uses a TorchQuantum ``QuantumDevice`` to encode
2×2 image patches via a random two‑qubit layer, measures the state, and then
evaluates a fixed quantum kernel (a sequence of Ry gates) on the resulting
measurement vectors.  An alias ``Kernel`` is provided for backward compatibility
with the original ``QuantumKernelMethod`` interface.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum analogue of the classical quanvolution filter.

    It encodes each 2×2 patch into a 4‑qubit register, applies a random layer,
    measures all qubits, and concatenates the results into a feature vector.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that applies Ry gates to each qubit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Random circuit to introduce non‑trivial correlations
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Concatenated measurement vector of shape (batch, 4 * 14 * 14).
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Stack the four pixels of the 2×2 patch
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class KernalAnsatz(tq.QuantumModule):
    """Fixed quantum kernel based on a sequence of Ry gates."""
    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Un‑encode y
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class HybridKernel(tq.QuantumModule):
    """Quantum hybrid kernel: quantum convolution followed by a quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.qfilter = QuantumQuanvolutionFilter()
        # Kernel ansatz: a simple sequence of Ry gates on each qubit
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value between two batches of images.
        """
        # Extract quantum features
        feat_x = self.qfilter(x)  # (batch, 4 * 14 * 14)
        feat_y = self.qfilter(y)
        # Use the same quantum device and ansatz to compute overlap
        self.ansatz(self.q_device, feat_x, feat_y)
        # Return the absolute value of the first amplitude as the kernel
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """
        Return Gram matrix for two batches of images.
        """
        return np.array([[self.forward(a[i:i+1], b[j:j+1]).item() for j in range(b.size(0))] for i in range(a.size(0))])


# Backward compatibility alias
Kernel = HybridKernel

__all__ = ["HybridKernel", "Kernel", "KernalAnsatz"]
