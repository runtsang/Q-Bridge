import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import Aer, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

class QuantumCircuitWrapper:
    """Parametrised circuit that returns the expectation of a Pauli observable."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Autograd wrapper that forwards a scalar through a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        theta_vals = inputs.detach().cpu().numpy()
        expectations = ctx.circuit.run(theta_vals)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for theta in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([theta + shift])
            left = ctx.circuit.run([theta - shift])
            grad_inputs.append((right - left) / 2.0)
        grad_inputs = torch.tensor(grad_inputs, dtype=grad_output.dtype,
                                   device=grad_output.device)
        return grad_inputs * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that maps a scalar to a quantum expectation value."""
    def __init__(self, n_qubits: int, backend, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class EstimatorQNN(nn.Module):
    """
    A hybrid estimator that blends a classical feed‑forward network with a
    quantum expectation head.  It can be used for regression or binary
    classification, and offers a pure classical mode for quick prototyping.
    """
    def __init__(self,
                 in_features: int = 2,
                 hidden_sizes: tuple[int,...] = (8, 4),
                 activation: nn.Module = nn.Tanh(),
                 shift: float = 0.0,
                 quantum: bool = False,
                 classification: bool = False,
                 n_qubits: int = 1,
                 backend=None,
                 shots: int = 100) -> None:
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            prev = h
        self.net = nn.Sequential(*layers)
        self.quantum = quantum
        self.classification = classification
        if quantum:
            if backend is None:
                backend = Aer.get_backend("aer_simulator")
            self.head = Hybrid(n_qubits, backend, shots, shift)
        else:
            self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        out = self.head(x)
        if self.classification:
            prob = torch.sigmoid(out)
            return torch.cat((prob, 1 - prob), dim=-1)
        return out

__all__ = ["EstimatorQNN", "QuantumCircuitWrapper", "HybridFunction", "Hybrid"]
