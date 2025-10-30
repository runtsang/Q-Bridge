"""
Quantum primitives for the hybrid classifier and kernel methods.

The module defines:
* `QuantumCircuit` – a TorchQuantum circuit that encodes a scalar input
  via an RY rotation, applies a random layer, and measures the Z
  expectation.
* `HybridFunction` – autograd wrapper that implements the parameter‑shift
  rule.
* `Hybrid` – a PyTorch module exposing the quantum circuit as a
  differentiable layer.
* `QuantumKernel` – a fixed circuit that implements a quantum kernel
  between two input vectors.
* `kernel_matrix` – helper that builds a Gram matrix using the quantum
  kernel.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumCircuit(tq.QuantumModule):
    """
    TorchQuantum module that maps a scalar input to a Z‑expectation value.
    The circuit consists of:
    * An RY rotation on all qubits with the input as the rotation angle.
    * A random layer of 30 single‑qubit gates drawn from the standard
      gate set.
    * Measurement of all qubits in the Z basis; the expectation is
      computed as the mean of the measurement outcomes.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        # Random layer for expressivity
        self.random_layer = tq.RandomLayer(n_ops=30,
                                          wires=list(range(self.n_qubits)))

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            1‑D tensor of shape (batch,) containing the rotation angle for
            each sample.  The same angle is applied to all qubits.
        """
        qdev.reset_states(x.shape[0])
        # Apply RY rotation with the same angle to all qubits
        self.ry(qdev, wires=list(range(self.n_qubits)), params=x)
        # Add random layer
        self.random_layer(qdev)
        # Measure all qubits in Z basis
        states = tq.MeasureAll(qdev, tq.PauliZ)
        # Expectation is the mean of the measurement outcomes
        # States are in {+1, -1}; convert to expectation value
        expectation = torch.mean(states, dim=1)
        qdev.states = expectation.unsqueeze(1)  # store for backward


class HybridFunction(torch.autograd.Function):
    """
    Wrapper that forwards a 1‑D tensor through a `QuantumCircuit` and
    returns the scalar expectation value.  Gradients are computed with
    the parameter‑shift rule.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Create a quantum device
        qdev = tq.QuantumDevice(n_wires=circuit.n_qubits, bsz=inputs.shape[0], device=inputs.device)
        # Forward pass
        circuit(qdev, inputs)
        # The device stores the expectation in its states attribute
        expectation = qdev.states.squeeze(-1)
        ctx.save_for_backward(inputs)
        return expectation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        # Parameter‑shift rule
        grad_inputs = torch.empty_like(inputs)
        for idx in range(inputs.shape[0]):
            x_plus = inputs.clone()
            x_minus = inputs.clone()
            x_plus[idx] += shift
            x_minus[idx] -= shift
            # Forward passes for shifted inputs
            qdev_plus = tq.QuantumDevice(n_wires=circuit.n_qubits,
                                         bsz=1,
                                         device=inputs.device)
            circuit(qdev_plus, x_plus[idx:idx+1])
            f_plus = qdev_plus.states.squeeze(-1)

            qdev_minus = tq.QuantumDevice(n_wires=circuit.n_qubits,
                                          bsz=1,
                                          device=inputs.device)
            circuit(qdev_minus, x_minus[idx:idx+1])
            f_minus = qdev_minus.states.squeeze(-1)

            grad_inputs[idx] = (f_plus - f_minus) / (2 * shift)
        return grad_inputs * grad_output, None, None


class Hybrid(tq.QuantumModule):
    """
    Differentiable quantum layer that can be composed with classical
    neural networks.
    """

    def __init__(self, n_qubits: int = 4, shift: float = math.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits)
        self.shift = shift

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        return HybridFunction.apply(x, self.circuit, self.shift)


class QuantumKernel(tq.QuantumModule):
    """
    Fixed quantum circuit that implements a kernel between two inputs.
    The kernel value is the absolute value of the inner product of the
    states prepared by two identical circuits with opposite phase
    rotations.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.ry = tq.RY(has_params=True, trainable=False)
        self.random_layer = tq.RandomLayer(n_ops=30,
                                          wires=list(range(self.n_qubits)))

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Compute the kernel value k(x, y) = |⟨ψ(x)|ψ(y)⟩|^2.
        """
        qdev.reset_states(x.shape[0])
        # Encode x
        self.ry(qdev, wires=list(range(self.n_qubits)), params=x)
        self.random_layer(qdev)
        # Copy state to a second device for y
        qdev_y = tq.QuantumDevice(n_wires=self.n_qubits,
                                  bsz=y.shape[0],
                                  device=x.device)
        self.ry(qdev_y, wires=list(range(self.n_qubits)), params=y)
        self.random_layer(qdev_y)
        # Compute overlap via swap test or inner product
        # Here we use the simple inner‑product estimator:
        # |⟨ψ(x)|ψ(y)⟩|^2 = |⟨ψ(x)|ψ(y)⟩|^2
        # For simplicity we approximate by measuring in computational basis
        # and taking the absolute value of the dot product of state vectors.
        states_x = qdev.states.squeeze(-1)
        states_y = qdev_y.states.squeeze(-1)
        kernel_val = torch.abs(torch.dot(states_x, states_y))
        qdev.states = kernel_val.unsqueeze(0)


def kernel_matrix(a: List[torch.Tensor], b: List[torch.Tensor]) -> np.ndarray:
    """
    Build a Gram matrix between two lists of 1‑D tensors using the
    quantum kernel.  The function returns a NumPy array of shape
    (len(a), len(b)).
    """
    kernel = QuantumKernel()
    gram = np.empty((len(a), len(b)), dtype=float)
    for i, xi in enumerate(a):
        for j, yj in enumerate(b):
            # The quantum kernel expects a batch of inputs; we provide
            # single‑sample batches.
            qdev = tq.QuantumDevice(n_wires=kernel.n_qubits,
                                    bsz=1,
                                    device=xi.device)
            kernel(qdev, xi, yj)
            gram[i, j] = qdev.states.squeeze().item()
    return gram


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid",
           "QuantumKernel", "kernel_matrix"]
