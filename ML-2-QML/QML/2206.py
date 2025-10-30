from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """
    Simple two‑qubit circuit with a single parameter per qubit.
    The parameter is expected to be a real number in [0, 2π].
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for each theta in thetas."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Expectation value of Z on first qubit
        exp = 0.0
        for state, cnt in counts.items():
            z = 1 if state[-1] == "0" else -1
            exp += z * cnt
        exp /= self.shots
        return np.array([exp])

class HybridFunctionQuantum(torch.autograd.Function):
    """Wraps the quantum circuit in a differentiable PyTorch function."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float = np.pi/2):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().flatten()
        expectation = circuit.run(thetas)
        output = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            exp_plus = ctx.circuit.run([val + shift])[0]
            exp_minus = ctx.circuit.run([val - shift])[0]
            grads.append(exp_plus - exp_minus)
        grad_inputs = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device) * grad_output
        return grad_inputs, None, None

class HybridEstimatorQNN(nn.Module):
    """
    Quantum‑enhanced estimator that replaces the final activation with a
    quantum expectation value.  The network structure mirrors the classical
    HybridEstimatorQNN but the head is now a quantum circuit.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, n_qubits: int = 2,
                 backend=None, shots: int = 100, shift: float = np.pi/2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.quantum_head = QuantumCircuitWrapper(n_qubits=n_qubits, backend=backend, shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return HybridFunctionQuantum.apply(out, self.quantum_head, self.shift)

__all__ = ["HybridEstimatorQNN", "HybridFunctionQuantum", "QuantumCircuitWrapper"]
