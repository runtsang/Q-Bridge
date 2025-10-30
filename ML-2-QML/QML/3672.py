"""Hybrid fully connected layer with quantum kernel ansatz."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumKernelAnsatz(tq.QuantumModule):
    """
    Implements a quantum kernel that encodes two data vectors into a single
    circuit via Ry rotations and their inverses and measures the overlap
    between the resulting states.
    """

    def __init__(self, n_wires: int, ref: torch.Tensor | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ref = ref or torch.zeros(n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> None:
        """
        Encode the input vector `x` with Ry gates, then encode the reference
        vector with negative Ry gates (effectively un‑rotating the state).
        """
        self.q_device.reset_states(1)
        for i in range(self.n_wires):
            param = x[i] if x.shape[0] > i else 0.0
            func_name_dict["ry"](self.q_device, wires=i, params=param)
        for i in range(self.n_wires):
            param = -self.ref[i]
            func_name_dict["ry"](self.q_device, wires=i, params=param)


class HybridFCL(tq.QuantumModule):
    """
    Quantum hybrid fully connected layer. The `run` method accepts an
    iterable of parameters, uses them as the input vector to the kernel
    ansatz and returns the expectation value of the first qubit.
    """

    def __init__(self, n_qubits: int = 4, ref: Sequence[float] | None = None) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.ref = torch.tensor(ref, dtype=torch.float32) if ref is not None else torch.zeros(n_qubits)
        self.ansatz = QuantumKernelAnsatz(n_qubits, self.ref)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Execute the quantum circuit and return the probability of the state |0...0>.
        """
        self.ansatz(thetas)
        probs = self.ansatz.q_device.states.view(-1)
        return torch.abs(probs[0])

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimic the classical FCL interface by converting an iterable of
        parameters into a tensor and returning the kernel evaluation as a
        NumPy array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).unsqueeze(0)
        return self.forward(values).detach().numpy().reshape(-1)


def FCL() -> HybridFCL:
    """Return an instance of the quantum hybrid fully‑connected layer."""
    return HybridFCL()


__all__ = ["HybridFCL", "FCL"]
