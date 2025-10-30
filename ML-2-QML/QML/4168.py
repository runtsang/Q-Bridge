"""Quantum‑augmented version of the CombinedConvNet."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import qiskit
from qiskit import assemble, transpile

# Import the classical backbone
from Conv__gen148 import ConvFilter, FraudLayerParameters, build_combined_program, CombinedConvNet

class QuantumCircuit:
    """Parametrised circuit that returns the expectation value of Z."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: th} for th in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            bits = np.array([int(b) for b in counts.keys()])
            return (bits * probs).sum()
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp = circuit.run(inputs.tolist())
        out = torch.tensor(exp, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for i, val in enumerate(inputs.numpy()):
            grads.append(
                ctx.circuit.run([val + shift[i]]) - ctx.circuit.run([val - shift[i]])
            )
        grad = torch.tensor(grads, dtype=torch.float32)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Layer that forwards activations through the quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class CombinedQuantumConvNet(CombinedConvNet):
    """Hybrid network that replaces the final linear head with a quantum expectation."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        fraud_params: Iterable[FraudLayerParameters] = (),
        n_qubits: int = 2,
    ) -> None:
        super().__init__(kernel_size, threshold, fraud_params, n_qubits)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum_head = Hybrid(n_qubits, backend, shots=200, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)          # (batch, 2)
        x = self.fraud_net(x)     # (batch, 1)
        x = self.linear(x)        # (batch, n_qubits)
        x = self.quantum_head(x)  # (batch, 1)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "CombinedQuantumConvNet"]
