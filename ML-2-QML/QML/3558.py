import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumExpectationLayer(nn.Module):
    """
    Forward pass through the parameterised circuit returning expectation values.
    """
    def __init__(self, n_qubits: int, shots: int = 100) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = QuantumCircuit.Parameter("theta")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: p} for p in params],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array([int(k, 2) for k in count_dict.keys()])
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        else:
            return np.array([expectation(result)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        exp_vals = self.run(x_np)
        return torch.tensor(exp_vals, device=x.device, dtype=torch.float32).unsqueeze(-1)

class HybridFunction(autograd.Function):
    """
    Differentiable wrapper using parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, layer: QuantumExpectationLayer, shift: float) -> torch.Tensor:
        ctx.layer = layer
        ctx.shift = shift
        with torch.no_grad():
            out = layer(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        x, = ctx.saved_tensors
        shift = ctx.shift
        layer = ctx.layer
        grad = torch.zeros_like(x)
        for i in range(x.size(1)):
            x_fwd = x.clone()
            x_fwd[:, i] += shift
            x_bwd = x.clone()
            x_bwd[:, i] -= shift
            with torch.no_grad():
                fwd = layer(x_fwd).detach()
                bwd = layer(x_bwd).detach()
            grad[:, i] = (fwd - bwd).squeeze(-1) / 2.0
        return grad * grad_output, None, None

class QuantumHybridLayer(nn.Module):
    """
    Wrapper combining QuantumExpectationLayer and HybridFunction to provide
    a fully differentiable quantum expectation head.
    """
    def __init__(self, n_qubits: int, shift: float, shots: int = 100, device: str = "cpu") -> None:
        super().__init__()
        self.quantum_layer = QuantumExpectationLayer(n_qubits, shots)
        self.shift = shift
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.quantum_layer, self.shift)
