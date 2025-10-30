"""
Hybrid CNN + Quantum + Kernel binary classifier – classical side.

The module implements:
* `HybridFunction` – an autograd function that forwards through a quantum
  circuit and computes gradients using the parameter‑shift rule.
* `Hybrid` – a thin wrapper around a `QuantumCircuit` that exposes a
  differentiable layer.
* `RBFKernel` – classical radial‑basis‑function kernel used for optional
  regularisation.
* `Kernel` – a simple wrapper that can use either the classical RBF kernel
  or a quantum kernel (imported from the QML side).
* `QCNet` – the full network: a 2‑D CNN backbone, a linear feature
  extractor, the hybrid quantum head, and an optional kernel head.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Quantum primitives are defined in the QML module.
# Import them here to keep the classical implementation independent.
try:
    from.qml_quantum import QuantumCircuit, quantum_kernel_matrix
except Exception:  # pragma: no cover
    # When the QML module is not available, provide stubs so that the
    # module can be imported for documentation or unit tests.
    class QuantumCircuit:  # type: ignore[misc]
        def __init__(self, *_, **__):  # pragma: no cover
            raise RuntimeError("QuantumCircuit requires the QML module.")
    def quantum_kernel_matrix(*_, **__):  # pragma: no cover
        raise RuntimeError("Quantum kernel requires the QML module.")


class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards a scalar input through a quantum
    circuit and returns its expectation value.  Gradients are computed
    with the parameter‑shift rule, which is exact for single‑parameter
    circuits and inexpensive to evaluate.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        """
        Forward pass: evaluate the quantum circuit on the given inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            1‑D tensor of shape (batch,) containing the parameters to the
            quantum circuit.
        circuit : QuantumCircuit
            TorchQuantum module that implements ``forward`` returning a
            scalar expectation value.
        shift : float
            The shift used in the parameter‑shift rule.
        """
        ctx.shift = shift
        ctx.circuit = circuit
        # The circuit expects a 1‑D tensor; ensure shape compatibility.
        expectation = circuit(inputs)
        ctx.save_for_backward(inputs)
        return expectation

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        # Parameter‑shift rule: (f(x+shift) - f(x-shift)) / 2
        grad_inputs = torch.empty_like(inputs)
        for idx in range(inputs.shape[0]):
            x_plus = inputs.clone()
            x_minus = inputs.clone()
            x_plus[idx] += shift
            x_minus[idx] -= shift
            f_plus = circuit(x_plus)
            f_minus = circuit(x_minus)
            grad_inputs[idx] = (f_plus - f_minus) / (2 * shift)
        return grad_inputs * grad_output, None, None


class Hybrid(nn.Module):
    """
    A PyTorch module that exposes a quantum circuit as a differentiable layer.
    """

    def __init__(self, n_qubits: int, shift: float = math.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


class RBFKernel(nn.Module):
    """
    Classical radial‑basis‑function kernel.

    Parameters
    ----------
    gamma : float, optional
        Width parameter of the RBF kernel.  Defaults to ``1.0``.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel value between two vectors.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of the same shape.
        """
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))


class Kernel(nn.Module):
    """
    Wrapper that can use either a classical RBF kernel or a quantum kernel.
    The quantum kernel is imported lazily from the QML module.
    """

    def __init__(self, gamma: float = 1.0, use_quantum: bool = False) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if use_quantum:
            # Quantum kernel is defined in the QML module.
            from.qml_quantum import QuantumKernel  # pylint: disable=import-outside-toplevel
            self.kernel = QuantumKernel()
        else:
            self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)


def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], use_quantum: bool = False) -> np.ndarray:
    """
    Compute a Gram matrix between two collections of vectors.

    Parameters
    ----------
    a, b : list of torch.Tensor
        Each element is a 1‑D tensor.
    use_quantum : bool, optional
        If ``True`` use the quantum kernel; otherwise use the classical RBF.
    """
    if use_quantum:
        return quantum_kernel_matrix(a, b)
    else:
        kernel = RBFKernel()
        return np.array([[kernel(x, y).item() for y in b] for x in a])


class QCNet(nn.Module):
    """
    Hybrid CNN + quantum head for binary classification.

    The network consists of:
    * A 2‑D CNN that extracts features from RGB images.
    * A linear layer that projects the flattened feature map to a
      1‑dimensional vector.
    * A quantum hybrid layer that maps the linear output to a
      probability via a parameter‑shift differentiable expectation.
    * An optional kernel head that concatenates the RBF kernel value
      between the linear output and a set of support vectors.
    """

    def __init__(self,
                 n_qubits: int = 4,
                 shift: float = math.pi / 2,
                 support_vectors: Optional[list[torch.Tensor]] = None,
                 kernel_gamma: float = 1.0) -> None:
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Linear feature extractor
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum hybrid head
        self.hybrid = Hybrid(n_qubits, shift)

        # Optional kernel head
        self.support_vectors = support_vectors
        self.kernel = None
        if support_vectors is not None:
            self.kernel = RBFKernel(gamma=kernel_gamma)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Linear layers
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)  # shape: (batch,)

        # Quantum hybrid head
        q_out = self.hybrid(x)  # shape: (batch,)

        # Combine with kernel head if present
        if self.support_vectors is not None:
            # Compute kernel values between batch and support vectors
            kernel_vals = torch.stack(
                [self.kernel(x[i], sv) for i in range(x.shape[0]) for sv in self.support_vectors]
            ).reshape(x.shape[0], -1)
            # Concatenate with quantum output
            out = torch.cat([q_out.unsqueeze(-1), kernel_vals], dim=-1)
            # Final linear layer to produce logits
            logits = nn.Linear(out.shape[-1], 1)(out).squeeze(-1)
        else:
            logits = q_out

        # Sigmoid to produce probabilities
        probs = torch.sigmoid(logits)
        return torch.cat([probs.unsqueeze(-1), 1 - probs.unsqueeze(-1)], dim=-1)


__all__ = ["HybridFunction", "Hybrid", "RBFKernel", "Kernel",
           "kernel_matrix", "QCNet"]
