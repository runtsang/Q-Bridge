"""Quantum implementation of the HybridClassifier.

This module provides a hybrid neural network that replaces the final
classical layer with a parameterised quantum circuit.  The circuit
supports an arbitrary number of qubits and uses a parameter‑shift
gradient estimator for back‑propagation.  It is fully compatible with
PyTorch autograd and can be used as a drop‑in replacement for the
classical head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

import qiskit
from qiskit import transpile, assemble, Aer
from qiskit.circuit import Parameter


class QuantumCircuit:
    """
    Parameterised quantum circuit that computes the expectation value
    of the Pauli‑Z operator on the first qubit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : qiskit.providers.Backend
        Qiskit backend used for execution.
    shots : int
        Number of shots per circuit execution.
    """

    def __init__(self, n_qubits: int, backend, shots: int = 1024) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        theta = Parameter("θ")
        # Simple entangling circuit
        for q in range(n_qubits):
            self._circuit.h(q)
        for q in range(n_qubits - 1):
            self._circuit.cx(q, q + 1)
        self._circuit.rx(theta, 0)
        self._circuit.measure_all()

        self.theta = theta
        self.backend = backend
        self.shots = shots

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameter values."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: float(p)} for p in params],
        )
        job = self.backend.run(qobj)
        result = job.result()
        # Expectation of Z on qubit 0
        expectation = np.array(
            [
                result.get_counts().get("0" * self._circuit.num_qubits, 0)
                / self.shots
                for _ in params
            ]
        )
        # Convert counts to ±1 expectation
        return 1 - 2 * expectation


class QuantumHybridFunction(torch.autograd.Function):
    """
    Autograd wrapper that calls the quantum circuit and implements
    the parameter‑shift rule for gradients.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Run the circuit for each input in the batch
        expectations = circuit.run(inputs.detach().cpu().numpy())
        return torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        inputs = grad_output.device
        # Parameter‑shift gradient: (f(x+shift) - f(x-shift)) / 2
        grad_inputs = torch.zeros_like(grad_output)
        for i, val in enumerate(grad_output):
            pos = circuit.run(np.array([val.item() + shift]))
            neg = circuit.run(np.array([val.item() - shift]))
            grad = (pos - neg) / 2.0
            grad_inputs[i] = grad * val
        return grad_inputs, None, None


class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the quantum circuit.
    backend : qiskit.providers.Backend
        Backend used for execution.
    shots : int
        Number of shots per execution.
    shift : float
        Shift value for the parameter‑shift gradient.
    """

    def __init__(
        self,
        n_qubits: int,
        backend,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure a 1‑D batch
        if inputs.ndim > 1:
            inputs = inputs.squeeze(-1)
        return QuantumHybridFunction.apply(inputs, self.circuit, self.shift)


class HybridClassifier(nn.Module):
    """
    Hybrid neural network that replaces the final classification layer
    with a quantum expectation head.

    Parameters
    ----------
    in_features : int
        Number of features from the preceding network.
    n_qubits : int
        Number of qubits in the quantum circuit.
    backend : qiskit.providers.Backend | None
        Backend used for execution.  If None, Aer simulator is used.
    shots : int
        Number of shots per circuit execution.
    shift : float
        Shift used for the parameter‑shift gradient.
    mode : Literal["binary", "multiclass"]
        Determines the activation applied to the quantum output.
    num_classes : int
        Number of output classes for multiclass classification.
    """

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 2,
        backend=None,
        shots: int = 1024,
        shift: float = np.pi / 2,
        mode: Literal["binary", "multiclass"] = "binary",
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if backend is None:
            backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)
        self.mode = mode
        self.num_classes = num_classes
        if mode == "multiclass":
            # Map quantum output to logits for each class
            self.logit_transform = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self.hybrid(x)
        if self.mode == "binary":
            probs = torch.sigmoid(q_out)
            return torch.cat((probs, 1 - probs), dim=-1)
        elif self.mode == "multiclass":
            logits = self.logit_transform(q_out)
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            raise ValueError(f"Unsupported mode {self.mode!r}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.hybrid.circuit._circuit.num_qubits}, "
            f"n_qubits={self.hybrid.circuit._circuit.num_qubits}, mode={self.mode}, "
            f"num_classes={self.num_classes})"
        )


__all__ = ["QuantumCircuit", "Hybrid", "QuantumHybridFunction", "HybridClassifier"]
