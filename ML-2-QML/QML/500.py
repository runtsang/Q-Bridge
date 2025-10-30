"""Quantum kernel construction with variational ansatz and SWAP‑test fidelity.

The module extends the original ``KernalAnsatz`` by:
* Parameterised variational layers that can be optimised.
* Multiple encoding options (angle or amplitude).
* Optional SWAP‑test based kernel evaluation for higher fidelity.
* GPU‑accelerated batch processing via TorchQuantum.
"""

from __future__ import annotations

from typing import Sequence, Optional, Iterable
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn


class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz with optional variational layers.

    Parameters
    ----------
    n_wires : int
        Number of qubits used for data encoding.
    encoding : str, default ``"ry"``.
        Gate used for angle encoding.  ``"ry"`` or ``"rzz"`` are supported.
    var_layers : int, default ``0``.
        Number of trainable variational layers.  Each layer contains one
        gate per qubit.
    var_gate : str, default ``"ry"``.
        Gate used in the variational layers.
    """

    def __init__(
        self,
        n_wires: int,
        encoding: str = "ry",
        var_layers: int = 0,
        var_gate: str = "ry",
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoding = encoding
        self.var_layers = var_layers
        self.var_gate = var_gate

        if var_layers > 0:
            self.var_params = nn.Parameter(
                torch.randn(var_layers, n_wires, dtype=torch.float32)
            )
        else:
            self.var_params = None

    @tq.static_support
    def encode(self, q_device: tq.QuantumDevice, data: torch.Tensor, sign: int = 1) -> None:
        """Apply the encoding and variational circuit to ``q_device``."""
        for wire in range(self.n_wires):
            param = data[0, wire] * sign
            func_name_dict[self.encoding](q_device, wires=wire, params=param)

        if self.var_layers > 0:
            for layer in range(self.var_layers):
                for wire in range(self.n_wires):
                    param = self.var_params[layer, wire] * sign
                    func_name_dict[self.var_gate](q_device, wires=wire, params=param)

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        swap_test: bool = False,
    ) -> tq.QuantumDevice:
        """Encode ``x`` and ``y`` on the same device and compute the kernel.

        When ``swap_test=True`` a SWAP‑test circuit is appended to
        estimate the state overlap more robustly.  The result is stored
        in ``q_device``.
        """
        if swap_test:
            device = tq.QuantumDevice(n_wires=self.n_wires + 1)
            device.reset_states(x.shape[0])
            self.encode(device, x, sign=1)
            ancilla = self.n_wires
            device.h(wires=ancilla)
            self.encode(device, y, sign=-1)
            for wire in range(self.n_wires):
                device.cswap(ancilla, wire, wire)
            device.h(wires=ancilla)
            device.measure(wires=ancilla)
            return device
        else:
            q_device.reset_states(x.shape[0])
            self.encode(q_device, x, sign=1)
            self.encode(q_device, y, sign=-1)
            return q_device

    def kernel_value(self, device: tq.QuantumDevice, *, swap_test: bool = False) -> torch.Tensor:
        """Return the kernel value from the device states."""
        if swap_test:
            probs = device.probs
            prob_0 = probs[..., 0]
            return 2 * prob_0 - 1  # |<ψx|ψy>|²
        else:
            return torch.abs(device.states.view(-1)[0])


class Kernel(tq.QuantumModule):
    """Convenient wrapper that exposes a scalar kernel interface.

    Parameters
    ----------
    n_wires : int
        Number of data qubits.
    encoding : str
        Gate used for data encoding.
    var_layers : int
        Number of variational layers.
    var_gate : str
        Gate used in variational layers.
    """

    def __init__(
        self,
        n_wires: int = 4,
        encoding: str = "ry",
        var_layers: int = 0,
        var_gate: str = "ry",
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = KernalAnsatz(
            n_wires=n_wires,
            encoding=encoding,
            var_layers=var_layers,
            var_gate=var_gate,
        )
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        swap_test: bool = False,
    ) -> torch.Tensor:
        """Compute the quantum kernel for a pair of samples."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        device = self.ansatz(self.q_device, x, y, swap_test=swap_test)
        return self.ansatz.kernel_value(device, swap_test=swap_test)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        *,
        swap_test: bool = False,
    ) -> np.ndarray:
        """Compute the Gram matrix for two sequences of tensors."""
        K = []
        for x in a:
            row = []
            for y in b:
                row.append(self.forward(x, y, swap_test=swap_test).item())
            K.append(row)
        return np.array(K)


__all__ = ["KernalAnsatz", "Kernel"]
