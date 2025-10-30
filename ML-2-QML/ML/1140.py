"""
Hybrid classical‑quantum convolution module.
Provides a trainable convolution followed by a quantum feature map
that can be differentiated using a custom autograd function.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

class QuantumExpectation(Function):
    """
    Autograd wrapper that evaluates a parameterised quantum circuit
    and returns the average probability of measuring |1> across all qubits.
    Gradient is computed via the parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, params: torch.Tensor, kernel_size: int, shots: int) -> torch.Tensor:
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.ry(theta[i], i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        bind = {theta[i]: float(params[i].item()) for i in range(n_qubits)}
        bound_qc = qc.bind_parameters(bind)
        simulator = AerSimulator()
        job = simulator.run(bound_qc, shots=shots)
        result = job.result()
        counts = result.get_counts(bound_qc)
        total_ones = sum(bitstring.count('1') * cnt for bitstring, cnt in counts.items())
        prob = total_ones / (shots * n_qubits)
        ctx.save_for_backward(params)
        ctx.kernel_size = kernel_size
        ctx.shots = shots
        return torch.tensor(prob, dtype=torch.float32, device=params.device)

    @staticmethod
    def backward(ctx, grad_output):
        params, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        shots = ctx.shots
        n_qubits = kernel_size ** 2
        shift = np.pi / 2
        grads = torch.zeros_like(params)
        for i in range(n_qubits):
            params_plus = params.clone()
            params_plus[i] += shift
            probs_plus = QuantumExpectation.forward(None, params_plus, kernel_size, shots)
            params_minus = params.clone()
            params_minus[i] -= shift
            probs_minus = QuantumExpectation.forward(None, params_minus, kernel_size, shots)
            grads[i] = (probs_plus - probs_minus) / (2 * shift)
        return grads * grad_output, None, None

class ConvEnhanced(nn.Module):
    """
    Hybrid convolutional layer that replaces the original Conv implementation.
    It consists of a trainable 2‑D convolution followed by a quantum
    feature‑map that outputs a scalar value.  The quantum part is
    differentiable via the custom ``QuantumExpectation`` autograd function.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        shots: int = 1024,
        trainable_qc: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.n_qubits = kernel_size ** 2
        if trainable_qc:
            self.q_params = nn.Parameter(torch.randn(self.n_qubits))
        else:
            self.register_buffer('q_params', torch.randn(self.n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the convolution, flattens the output
        and feeds it through the quantum circuit.  Returns a scalar
        for each sample in the batch.
        """
        out = self.conv(x)
        out_flat = out.view(out.shape[0], -1)
        angles = out_flat * self.q_params
        probs = []
        for sample in angles:
            prob = QuantumExpectation.apply(sample, self.kernel_size, self.shots)
            probs.append(prob)
        return torch.stack(probs)

    def quantum_forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Directly evaluate the quantum circuit for a given parameter vector.
        """
        return QuantumExpectation.apply(params, self.kernel_size, self.shots)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, shots={self.shots})"

__all__ = ["ConvEnhanced"]
