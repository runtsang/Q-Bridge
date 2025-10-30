"""Hybrid quantum kernel combining a parameter‑free encoding and a random
variational layer.  The kernel evaluates the overlap of two quantum states
prepared from classical vectors using a fixed circuit topology.  The design
mirrors the classical RBF kernel but replaces the Euclidean distance with a
quantum distance that can capture non‑linear correlations via entanglement.

The module exposes a ``kernel_matrix`` helper that builds the Gram matrix
between two collections of feature vectors, making it drop‑in compatible with
classical kernel methods.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class HybridKernel(tq.QuantumModule):
    """Quantum kernel with a fixed encoding and a random layer.

    The circuit consists of:
    * a deterministic encoding of the input vector into the ``ry`` gate of
      each qubit,
    * a random variational layer of two‑qubit gates,
    * a measurement of all qubits in the Pauli‑Z basis.

    The kernel value is the absolute overlap of the two encoded states.
    """
    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoding: one ry gate per input feature
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def _apply_ops(self, qdev: tq.QuantumDevice, data: torch.Tensor,
                   reverse: bool = False) -> None:
        """Apply the encoding and variational layer to ``data``."""
        if reverse:
            # Apply the encoding with negative parameters to cancel the
            # effect of the forward encoding.  This mimics the swap‑test
            # style overlap used in the classical kernel.
            params = -data
        else:
            params = data
        self.encoder(qdev, params)
        self.q_layer(qdev)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value ``k(x, y)``.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of length ``n_wires`` representing two data points.
        """
        qdev = tq.QuantumDevice(self.n_wires, bsz=1)
        # Encode x
        self._apply_ops(qdev, x, reverse=False)
        # Encode -y (reverse) to compute overlap
        self._apply_ops(qdev, y, reverse=True)
        measurement = self.measure(qdev)
        # The first element of the state vector corresponds to |00...0>
        return torch.abs(measurement.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Return the Gram matrix between two collections of tensors.

    Parameters
    ----------
    a, b : sequences of torch.Tensor
        Each element is a 1‑D tensor of length ``n_wires``.
    """
    kernel = HybridKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridKernel", "kernel_matrix"]
