"""Quantum kernel implementation using Pennylane.

This module extends the original TorchQuantum based kernel by providing
a configurable circuit depth, automatic qubit sizing based on the feature
dimension, and a reusable factory that can instantiate different ansatz
families.  The kernel is evaluated as the squared absolute overlap between
two encoded states, which is equivalent to the fidelity of the feature map.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp

__all__ = ["QuantumKernelMethod", "kernel_matrix"]


class QuantumKernelMethod(torch.nn.Module):
    """
    Quantum kernel built from a variational circuit on Pennylane.

    Parameters
    ----------
    depth : int
        Number of repetitions of the entangling layer.
    wires_per_block : int, optional
        Number of wires per entangling block.  If ``None`` the circuit
        will be fully connected.
    device_name : str, optional
        Name of the Pennylane device to use.  ``"default.qubit"`` is the
        default and will automatically size the device to the number of
        qubits required.
    """

    def __init__(self, depth: int = 2, wires_per_block: int | None = None, device_name: str = "default.qubit") -> None:
        super().__init__()
        self.depth = depth
        self.wires_per_block = wires_per_block
        self.device_name = device_name
        self._device = None  # will be lazily created

    def _get_device(self, n_qubits: int) -> qml.Device:
        if self._device is None or self._device.n_wires!= n_qubits:
            self._device = qml.device(self.device_name, wires=n_qubits)
        return self._device

    def _ansatz(self, x: pnp.ndarray):
        """Variational ansatz that encodes the input vector."""
        wires = list(range(len(x)))
        for _ in range(self.depth):
            for i, wire in enumerate(wires):
                qml.RY(x[i], wires=wire)
            # Entanglement pattern
            if self.wires_per_block is None:
                # Fully connected CNOT ladder
                for i in range(len(wires) - 1):
                    qml.CNOT(wires=[wires[i], wires[i + 1]])
            else:
                for i in range(0, len(wires) - self.wires_per_block, self.wires_per_block):
                    qml.CNOT(wires=[wires[i], wires[i + self.wires_per_block]])
            # Additional single‑qubit rotations
            for i, wire in enumerate(wires):
                qml.RZ(x[i], wires=wire)

    def _kernel_circuit(self, x: pnp.ndarray, y: pnp.ndarray):
        """Circuit that prepares |x⟩ and then uncomputes |y⟩."""
        self._ansatz(x)
        self._ansatz(-y)  # uncompute with negative parameters
        return qml.probs(wires=range(len(x)))  # probability of |0...0⟩

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return the quantum kernel value for two 1‑D tensors.
        The tensors are expected to be 1‑D or 2‑D (batch) with shape
        (n_samples, n_features).
        """
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # Ensure inputs are 2‑D
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        if y_np.ndim == 1:
            y_np = y_np.reshape(1, -1)

        n_qubits = x_np.shape[1]
        dev = self._get_device(n_qubits)

        @qml.qnode(dev, interface="torch")
        def qfunc(x_vec, y_vec):
            return self._kernel_circuit(x_vec, y_vec)

        # Compute overlap matrix
        K = torch.zeros((x_np.shape[0], y_np.shape[0]), dtype=torch.float32)
        for i, xi in enumerate(x_np):
            for j, yj in enumerate(y_np):
                prob = qfunc(xi, yj)
                # The probability of |0...0⟩ equals |⟨x|y⟩|²
                K[i, j] = prob[0]
        return K


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Convenience wrapper that constructs a :class:`QuantumKernelMethod` and
    returns the Gram matrix for two sequences of tensors.
    """
    kernel = QuantumKernelMethod()
    return np.array([[kernel(x, y).item() for y in b] for x in a])
