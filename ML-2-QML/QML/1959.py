"""Quantum convolutional filter (Quanvolution) implemented with PennyLane.

The class implements a variational quantum circuit that acts as a
convolutional filter over 2D data. Each kernel size corresponds to a
separate quantum node. The circuit encodes classical data via Ry
rotations, applies a learnable rotation layer, and measures the
expectation of the Pauli‑Z operator on all qubits. A shared classical
threshold is used for post‑processing, mirroring the classical
module.

Example usage:

    conv_q = ConvGen154(kernel_sizes=[2,3], threshold=0.5)
    out = conv_q(torch.randn(1,1,32,32))
"""

import pennylane as qml
import numpy as np
import torch
from torch import nn
from typing import List, Union

class ConvGen154(nn.Module):
    """
    Quantum convolutional filter that shares a learnable threshold
    across all kernels. Each kernel size maps to a separate variational
    quantum circuit.
    """
    def __init__(
        self,
        kernel_sizes: Union[int, List[int]] = 2,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))

        # Create a quantum node for each kernel size
        self.qnodes = nn.ModuleList()
        for k in self.kernel_sizes:
            n_qubits = k * k
            dev = qml.device("default.qubit", wires=n_qubits)

            # Learnable weights for the rotation layer
            weight = nn.Parameter(torch.randn(n_qubits))

            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(x: torch.Tensor):
                # Encode classical data via Ry rotations
                for i in range(n_qubits):
                    qml.RY(x[i] * np.pi, wires=i)
                # Variational rotation layer
                for i in range(n_qubits):
                    qml.RY(weight[i], wires=i)
                # Expectation value of the first qubit
                return qml.expval(qml.PauliZ(0))

            self.qnodes.append(circuit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Concatenated outputs from all quantum kernels.
            Shape: (batch, len(kernel_sizes), H', W')
        """
        batch, _, h, w = x.shape
        feats = []

        for idx, k in enumerate(self.kernel_sizes):
            out_h = h - k + 1
            out_w = w - k + 1
            conv_out = torch.zeros(batch, out_h, out_w, device=x.device)
            for i in range(out_h):
                for j in range(out_w):
                    patch = x[:, :, i:i + k, j:j + k]
                    patch_flat = patch.view(batch, -1)
                    for b in range(batch):
                        qval = self.qnodes[idx](patch_flat[b])
                        conv_out[b, i, j] = qval
            conv_out = torch.sigmoid(conv_out - self.threshold)
            feats.append(conv_out)

        return torch.stack(feats, dim=1)

__all__ = ["ConvGen154"]
