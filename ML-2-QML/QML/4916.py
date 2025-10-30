"""Hybrid quantum kernel that extends a quantum RBF by embedding data
with a patchwise encoder and a quantum self‑attention style circuit."""

import torch
import torchquantum as tq
import numpy as np
from typing import Sequence
from torchquantum.functional import crx

class HybridKernelMethod(tq.QuantumModule):
    """Quantum kernel that combines a patch‑wise quantum encoder,
    a random layer, and a parameterised attention‑style CRX block.
    The module is fully quantum and can be used as a drop‑in
    replacement for the classical ``Kernel`` class.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits used for each patch (patch_size²).
    attention_depth : int, default 2
        Number of attention‑style CRX layers applied after encoding.
    """

    def __init__(self, n_wires: int = 4, attention_depth: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.attention_depth = attention_depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # patch encoder: Ry rotations on each qubit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(self.n_wires)
            ]
        )
        # random layer to mix information
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        # attention‑style CRX gates
        self.attention_ops = [
            {"func": "crx", "wires": [i, (i + 1) % self.n_wires]}
            for i in range(self.attention_depth)
        ]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _apply_attention(self, q_device: tq.QuantumDevice, params: torch.Tensor) -> None:
        """Apply a sequence of parameterised CRX gates."""
        for i, op in enumerate(self.attention_ops):
            idx = i % params.shape[1]
            crx(q_device, op["wires"][0], op["wires"][1], params[:, idx])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap between the encoded states of ``x`` and ``y``."""
        # encode inputs
        bs = x.shape[0]
        self.q_device.reset_states(bs)
        self.encoder(self.q_device, x)
        self.random_layer(self.q_device)
        # use deterministic parameters for reproducibility
        params = torch.zeros(bs, self.attention_depth)
        self._apply_attention(self.q_device, params)
        psi_x = self.q_device.states.view(-1)[0].clone()

        # encode second input
        self.q_device.reset_states(bs)
        self.encoder(self.q_device, y)
        self.random_layer(self.q_device)
        self._apply_attention(self.q_device, params)
        psi_y = self.q_device.states.view(-1)[0].clone()

        # compute overlap magnitude
        overlap = torch.abs(psi_x.conj() @ psi_y)
        return overlap

    @staticmethod
    def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix using the quantum kernel."""
        model = HybridKernelMethod()
        torch.set_grad_enabled(False)
        gram = torch.zeros(len(a), len(b))
        for i, xi in enumerate(a):
            xi = xi.unsqueeze(0)
            for j, yj in enumerate(b):
                yj = yj.unsqueeze(0)
                gram[i, j] = model(xi, yj).item()
        return gram.numpy()

__all__ = ["HybridKernelMethod"]
