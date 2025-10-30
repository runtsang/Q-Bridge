"""Quantum kernel module built on TorchQuantum.  The implementation follows
the same public API as :class:`QuantumKernelMethod` in the classical module,
allowing a drop‑in switch between back‑ends.

Key design choices
------------------
* An adaptive variational ansatz composed of a RandomLayer followed by a
  trainable RX/RZ sequence.  The depth of the RandomLayer is a
  hyper‑parameter that controls expressivity.
* Data encoding is performed via Ry rotations on each qubit.
* The kernel value is the absolute value of the overlap between two
  encoded states, estimated from a single‑shot measurement of Pauli‑Z
  on all wires.  This measurement is differentiable thanks to
  TorchQuantum's automatic differentiation support.
"""

from __future__ import annotations

import torch
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumKernelMethod(tq.QuantumModule):
    """
    Variational quantum kernel with a learnable ansatz depth.
    """

    def __init__(self, n_wires: int = 4, depth: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device="cpu")
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self) -> tq.QuantumModule:
        """
        Construct a parameterised circuit consisting of a RandomLayer
        followed by a fixed trainable sequence of single‑qubit rotations.
        """
        class _Ansatz(tq.QuantumModule):
            def __init__(self, n_wires, depth):
                super().__init__()
                self.n_wires = n_wires
                self.random = tq.RandomLayer(n_ops=10 * depth,
                                             wires=list(range(n_wires)))
                self.rx = tq.RX(has_params=True, trainable=True)
                self.rz = tq.RZ(has_params=True, trainable=True)

            @tq.static_support
            def forward(self, qdev: tq.QuantumDevice) -> None:
                self.random(qdev)
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.rz(qdev, wires=w)

        return _Ansatz(self.n_wires, self.depth)

    @staticmethod
    def _encode(qdev: tq.QuantumDevice, vector: torch.Tensor) -> None:
        """Encode a classical vector via Ry rotations."""
        for idx, val in enumerate(vector):
            tqf.ry(qdev, wires=idx, params=val)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between two batches of vectors.
        The implementation is fully differentiable and can be used
        inside larger neural networks.
        """
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        batch = x.shape[0]
        kernel = torch.empty(batch, batch, device=x.device, dtype=x.dtype)

        for i, vec_x in enumerate(x):
            for j, vec_y in enumerate(y):
                # Prepare device
                self.q_device.reset_states(1)
                self._encode(self.q_device, vec_x)
                self.ansatz(self.q_device)

                state_x = self.q_device.states.squeeze().clone()

                self.q_device.reset_states(1)
                self._encode(self.q_device, vec_y)
                self.ansatz(self.q_device)

                state_y = self.q_device.states.squeeze()

                # Overlap as absolute square of inner product
                overlap = torch.abs(torch.dot(state_x, state_y.conj())) ** 2
                kernel[i, j] = overlap

        return kernel

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning the Gram matrix."""
        return self.forward(a, b)


__all__ = ["QuantumKernelMethod"]
