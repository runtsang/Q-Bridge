import numpy as np
import torch
import torch.nn as nn
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.providers.aer import AerSimulator
from qiskit.primitives import Sampler as QiskitSampler
from torch.autograd import Function

class QuantumSamplerCircuit:
    """Two‑qubit variational circuit used for sampling."""
    def __init__(self):
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)
        self.sampler = QiskitSampler(AerSimulator())
    def run(self, input_vals, weight_vals):
        bind_dict = {p: v for p, v in zip(self.input_params, input_vals)}
        bind_dict.update({p: v for p, v in zip(self.weight_params, weight_vals)})
        bound_circuit = self.circuit.bind_parameters(bind_dict)
        result = self.sampler.run(bound_circuit)
        statevector = result.get_statevector()
        probs = np.abs(statevector)**2
        return probs[0], probs[1]

class HybridFunction(Function):
    """Differentiable interface between PyTorch and the quantum sampler."""
    @staticmethod
    def forward(ctx, inputs, circuit: QuantumSamplerCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        probs = []
        for inp in inputs:
            inp_vals = inp[:2].tolist()
            weight_vals = inp[2:].tolist()
            p0, p1 = circuit.run(inp_vals, weight_vals)
            probs.append([p0, p1])
        probs_np = np.array(probs)
        probs_tensor = torch.tensor(probs_np, dtype=torch.float32)
        ctx.save_for_backward(inputs, probs_tensor)
        return probs_tensor
    @staticmethod
    def backward(ctx, grad_output):
        # No gradient back‑propagation to quantum parameters for simplicity
        return None, None, None

class SamplerQNNHybrid(nn.Module):
    """Quantum sampler network that outputs a probability distribution over 2 classes."""
    def __init__(self, shift=np.pi/2):
        super().__init__()
        self.circuit = QuantumSamplerCircuit()
        self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, 4) where first 2 are input params, last 2 are weight params
        return HybridFunction.apply(inputs, self.circuit, self.shift)

def SamplerQNN() -> SamplerQNNHybrid:
    return SamplerQNNHybrid()
