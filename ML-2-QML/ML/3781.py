import torch
import torch.nn as nn
import numpy as np

class FullyConnectedLayer(nn.Module):
    """
    Classical dense layer that emulates the quantum expectation
    of a set of angles.  It can be dropped in as a stand‑in for the
    original FCL example.
    """
    def __init__(self, n_features: int = 1, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equivalent to the original `run` method: tanh(mean(linear(x)))
        values = self.linear(x)
        exp_val = torch.tanh(values).mean(dim=0, keepdim=True)
        return exp_val

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and a quantum circuit.
    If `circuit` is None, falls back to a sigmoid activation.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit, shift: float = 0.0):
        ctx.shift = shift
        ctx.circuit = circuit
        if circuit is None:
            out = torch.sigmoid(inputs + shift)
            ctx.save_for_backward(out)
            return out
        # inputs is a 2‑D tensor of angles (batch, n_qubits*depth)
        theta = inputs.detach().cpu().numpy()
        expectation = circuit.run(theta).astype(np.float32)
        out = torch.tensor(expectation, device=inputs.device)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, out = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        if circuit is None:
            grad = grad_output * out * (1 - out)
            return grad, None, None
        # Finite‑difference approximation
        h = 1e-4
        theta = inputs.detach().cpu().numpy()
        theta_plus = theta + h
        theta_minus = theta - h
        exp_plus = circuit.run(theta_plus).astype(np.float32)
        exp_minus = circuit.run(theta_minus).astype(np.float32)
        grad = (exp_plus - exp_minus) / (2 * h)
        grad = torch.tensor(grad, device=inputs.device)
        return grad * grad_output, None, None

class QuantumFullyConnectedHybrid(nn.Module):
    """
    Hybrid fully‑connected layer that can operate in two modes:
    1. Classical‑only: linear transform followed by a sigmoid.
    2. Quantum‑augmented: passes the linear outputs through a
       parameterised quantum circuit and uses its expectation
       value as the activation.
    """
    def __init__(self,
                 n_features: int = 1,
                 use_quantum: bool = False,
                 circuit: object = None,
                 shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.use_quantum = use_quantum
        self.circuit = circuit
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)  # shape (batch, 1)
        if self.use_quantum and self.circuit is not None:
            # Expand each linear output to a vector of angles for the circuit
            n_params = self.circuit.n_qubits * self.circuit.depth
            angles = linear_out.repeat_interleave(n_params, dim=0)
            angles = angles.view(-1, n_params)
            out = HybridFunction.apply(angles, self.circuit, self.shift)
        else:
            out = torch.sigmoid(linear_out.squeeze() + self.shift)
        return out

__all__ = ["FullyConnectedLayer", "HybridFunction", "QuantumFullyConnectedHybrid"]
