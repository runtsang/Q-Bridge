"""Hybrid quantum kernel using a TorchQuantum ansatz.

The quantum kernel reproduces the behaviour of the original
`QuantumKernelMethod` but replaces the fixed Ry ansatz with a
parameterised circuit inspired by the SamplerQNN example.  The
kernel is the absolute overlap of the two encoded states.
"""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["HybridKernelMethod"]


class _SamplerAnsatz(tq.QuantumModule):
    """Parameterised 2‑qubit circuit (Ry–CX–Ry blocks)."""
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # 4 trainable Ry parameters as in the original SamplerQNN
        self.params = torch.nn.Parameter(torch.randn(4))

    def forward(self, q_device: tq.QuantumDevice, inputs: torch.Tensor) -> None:
        """Encode the classical inputs and apply trainable rotations."""
        q_device.reset_states(inputs.shape[0])

        # Encode inputs
        for i in range(2):
            func_name_dict["ry"](q_device, wires=[i], params=inputs[:, i:i+1])

        # Entanglement and trainable Ry layers
        func_name_dict["cx"](q_device, wires=[0, 1])
        for i in range(2):
            func_name_dict["ry"](q_device, wires=[i], params=self.params[i:i+1])
        func_name_dict["cx"](q_device, wires=[0, 1])
        for i in range(2):
            func_name_dict["ry"](q_device, wires=[i], params=self.params[2+i:3+i])


class HybridKernelMethod(tq.QuantumModule):
    """Quantum kernel that computes the overlap of two encoded states.

    The kernel is defined as ``|⟨ψ(x)|ψ(y)⟩|`` where each state is prepared
    by the :class:`_SamplerAnsatz`.  This mirrors the behaviour of the
    classical RBF kernel but in a quantum Hilbert space.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = _SamplerAnsatz()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value for a single pair of samples."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        # Prepare state for x
        self.ansatz(self.q_device, x)
        state_x = self.q_device.states.clone()

        # Prepare state for y on a fresh device
        self.q_device.reset_states(y.shape[0])
        self.ansatz(self.q_device, y)
        state_y = self.q_device.states.clone()

        # Overlap of the two states
        return torch.abs(torch.sum(state_x * state_y.conj(), dim=-1)).squeeze()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the full Gram matrix between two sets of samples."""
        return np.array(
            [[self.forward(x.unsqueeze(0), y.unsqueeze(0)).item() for y in b] for x in a]
        )
