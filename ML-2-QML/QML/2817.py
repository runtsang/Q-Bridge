"""Quantum implementation of the hybrid sampler.

The class ``HybridSamplerQNN`` can operate in two modes:

* ``sampler`` – evaluates a parameterised circuit and returns
  measurement probabilities of the first two computational basis
  states.
* ``kernel`` – implements a quantum kernel based on a programmable
  ansatz that mirrors the classical RBF kernel.

The implementation is intentionally lightweight and uses only
torchquantum primitives.
"""

from __future__ import annotations

import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["HybridSamplerQNN", "SamplerQNN"]

# ----------------------------------------------------------------------
# Quantum kernel components – adapted from the QML seed
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encode two inputs into a quantum state and un‑encode the second."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the absolute amplitude of the |0000> state
        return torch.abs(self.q_device.states.view(-1)[0])

# ----------------------------------------------------------------------
# Hybrid sampler – combines a simple parametric circuit with the kernel
# ----------------------------------------------------------------------
class HybridSamplerQNN(tq.QuantumModule):
    """Hybrid quantum sampler that can act as a kernel or a probability sampler.

    Parameters
    ----------
    mode : str, default "sampler"
        One of ``"sampler"`` or ``"kernel"``.
    """

    def __init__(self, mode: str = "sampler") -> None:
        super().__init__()
        if mode not in {"sampler", "kernel"}:
            raise ValueError("mode must be either'sampler' or 'kernel'")
        self.mode = mode
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Simple circuit used for sampling
        self.sampler_ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
                {"input_idx": [0], "func": "cx", "wires": [0, 1]},
                {"input_idx": [1], "func": "cx", "wires": [1, 2]},
                {"input_idx": [2], "func": "cx", "wires": [2, 3]},
            ]
        )

        # Quantum kernel instance
        self._kernel = Kernel()

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """
        Operate in either sampler or kernel mode.

        In ``sampler`` mode ``y`` is ignored and the circuit is run
        with the parameters in ``x``.  The returned tensor contains
        the probabilities of the first two basis states.

        In ``kernel`` mode ``y`` must be provided.  The kernel value
        is computed as the overlap between the states encoded by
        ``x`` and ``y``.
        """
        if self.mode == "sampler":
            # Sampling: only x is used
            self.q_device.reset_states(x.shape[0])
            self.sampler_ansatz(self.q_device, x, torch.zeros_like(x))
            probs = self.q_device.probs
            # Return probabilities for the first two basis states
            return probs[:, :2]
        else:  # kernel mode
            if y is None:
                raise ValueError("y must be provided in kernel mode")
            return self._kernel(x, y)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return the Gram matrix between two sets of samples."""
        return self._kernel(a, b)

def SamplerQNN() -> HybridSamplerQNN:
    """Factory compatible with the original API – defaults to sampler mode."""
    return HybridSamplerQNN(mode="sampler")
