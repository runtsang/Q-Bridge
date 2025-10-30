from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit import Aer
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# 1. Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """
    Two‑qubit parameterised circuit that returns the expectation value of
    the Pauli‑Z operator. The circuit is a simple H‑RY‑measure pattern,
    which is easy to differentiate and simulate.
    """
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
        """Execute the circuit for each theta in `thetas`."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
# 2. Autograd bridge
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards inputs to a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.inputs = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(ctx.inputs)
        return torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Finite‑difference approximation of the gradient
        eps = 1e-3
        grads = []
        for i in range(ctx.inputs.shape[0]):
            theta_plus = ctx.inputs.copy()
            theta_minus = ctx.inputs.copy()
            theta_plus[i] += eps
            theta_minus[i] -= eps
            exp_plus = ctx.circuit.run(theta_plus)
            exp_minus = ctx.circuit.run(theta_minus)
            grads.append((exp_plus - exp_minus) / (2 * eps))
        grad_tensor = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grad_tensor * grad_output, None, None

# --------------------------------------------------------------------------- #
# 3. Hybrid quantum head
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """Quantum expectation head that can be attached to a classical network."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
# 4. Convenience factory
# --------------------------------------------------------------------------- #
def get_quantum_head(n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2) -> nn.Module:
    """
    Return a ready‑to‑attach quantum head.
    """
    backend = Aer.get_backend("qasm_simulator")
    return Hybrid(n_qubits, backend, shots, shift)

__all__ = [
    "QuantumCircuit",
    "HybridFunction",
    "Hybrid",
    "get_quantum_head",
]
