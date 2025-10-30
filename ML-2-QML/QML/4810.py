from __future__ import annotations

import numpy as np
import torch
from qiskit.circuit import QuantumCircuit as QiskitCircuit
from qiskit import assemble, transpile
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QuantumCircuitWrapper:
    """Parametrised circuit executed on a Qiskit backend."""
    def __init__(self, circuit: QiskitCircuit, backend, shots: int):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.circuit.parameters[i]: theta} for i, theta in enumerate(thetas)],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunctionQuantum(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.numpy())
        result = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.numpy():
            exp_plus = ctx.circuit.run(np.array([val + shift]))
            exp_minus = ctx.circuit.run(np.array([val - shift]))
            grads.append(exp_plus - exp_minus)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class HybridQuantum(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, circuit: QiskitCircuit, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit_wrapper = QuantumCircuitWrapper(circuit, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunctionQuantum.apply(inputs, self.circuit_wrapper, self.shift)

__all__ = [
    "QuantumCircuitWrapper",
    "HybridFunctionQuantum",
    "HybridQuantum",
]
