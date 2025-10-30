"""
Hybrid quantum kernel module that extends the original TorchQuantum
implementation with a variational feature map, a random layer, and
an optional attention‑style entanglement block.  The module can be used
directly in a kernel‑SVM or as a similarity measure for downstream
tasks.
"""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence, List


# --------------------------------------------------------------------------- #
# 1. Feature‑map ansatz: Ry rotations per input dimension followed by
#    a random layer that mixes the wires.
# --------------------------------------------------------------------------- #
class FeatureMap(tq.QuantumModule):
    """
    Encodes a classical vector into a quantum state using Ry rotations
    followed by a random circuit.  The design mirrors the
    Quanvolution encoder but operates on a flat vector.
    """

    def __init__(self, n_wires: int, n_random_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=n_random_ops, wires=list(range(n_wires)))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        """
        Encode a batch of vectors x into the quantum device.
        """
        q_device.reset_states(x.shape[0])
        self.encoder(q_device, x)
        self.random_layer(q_device)


# --------------------------------------------------------------------------- #
# 2. Attention‑style entanglement: a trainable sequence of rotations
#    followed by controlled‑RX gates that act like a lightweight
#    self‑attention mechanism.
# --------------------------------------------------------------------------- #
class AttentionLayer(tq.QuantumModule):
    """
    Adds a parameterized rotation on each qubit and a chain of
    controlled‑RX gates.  The parameters are stored in a torch
    ParameterTensor that can be learned if the module is wrapped
    in a higher‑level variational circuit.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # rotation parameters per qubit
        self.rotation_params = torch.nn.Parameter(
            torch.randn(n_wires * 3)  # rx, ry, rz per qubit
        )
        # entanglement parameters per adjacent pair
        self.entangle_params = torch.nn.Parameter(
            torch.randn(n_wires - 1)
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice) -> None:
        for i in range(self.n_wires):
            idx = 3 * i
            func_name_dict["rx"](q_device, wires=[i], params=self.rotation_params[idx])
            func_name_dict["ry"](q_device, wires=[i], params=self.rotation_params[idx + 1])
            func_name_dict["rz"](q_device, wires=[i], params=self.rotation_params[idx + 2])

        for i in range(self.n_wires - 1):
            func_name_dict["crx"](q_device, wires=[i, i + 1], params=self.entangle_params[i])


# --------------------------------------------------------------------------- #
# 3. Hybrid kernel module that evaluates the overlap of two feature maps
# --------------------------------------------------------------------------- #
class HybridKernelMethod(tq.QuantumModule):
    """
    Evaluates a quantum kernel that is a product of:
        1) a feature‑map ansatz (Ry + random layer)
        2) an attention‑style entanglement block
        3) an overlap measurement (absolute value of the first amplitude)
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.feature_map = FeatureMap(n_wires=self.n_wires)
        self.attention = AttentionLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel for a single pair (x, y).
        x, y: (batch, n_wires) tensors
        """
        # encode both vectors sequentially
        self.feature_map(self.q_device, x)
        self.attention(self.q_device)
        # reset and encode the second vector with a negative sign (swap test style)
        self.q_device.reset_states(x.shape[0])
        self.feature_map(self.q_device, -y)
        self.attention(self.q_device)
        # overlap via measurement
        amp = self.q_device.states.view(-1)[0]
        return torch.abs(amp)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Wrapper that reshapes inputs and returns a (batch, 1) kernel
        matrix element.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return self.kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Compute the full Gram matrix between two collections of samples.
        """
        return torch.stack(
            [
                torch.stack([self.forward(x, y) for y in b]).squeeze(-1)
                for x in a
            ]
        )


__all__ = [
    "FeatureMap",
    "AttentionLayer",
    "HybridKernelMethod",
]
