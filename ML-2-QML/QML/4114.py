import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

# ----------------------------------------------------------------------
# Classical components reused from the ML version
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool = True) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Quantum circuit wrapper
# ----------------------------------------------------------------------
class QuantumCircuitWrapper:
    """Two‑qubit variational circuit that returns the expectation of Z."""
    def __init__(self, shots: int = 100) -> None:
        self.circuit = qiskit.QuantumCircuit(2)
        self.circuit.h([0, 1])
        self.circuit.barrier()
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.ry(self.theta, [0, 1])
        self.circuit.measure_all()
        self.backend = AerSimulator()
        self.shots = shots

    def run(self, theta_vals: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: val} for val in theta_vals],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

# ----------------------------------------------------------------------
# Hybrid function bridging PyTorch and the quantum circuit
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit."""
    @staticmethod
    def forward(ctx, x: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        theta = x.detach().cpu().numpy()
        exp_vals = ctx.circuit.run(theta)
        out = torch.tensor(exp_vals, dtype=torch.float32)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, _ = ctx.saved_tensors
        shift = np.ones_like(x.numpy()) * ctx.shift
        grads = []
        for val in x.numpy():
            right = ctx.circuit.run([val + shift[0]])
            left = ctx.circuit.run([val - shift[0]])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class HybridLayer(nn.Module):
    """Layer that forwards a vector through the quantum circuit."""
    def __init__(self, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

# ----------------------------------------------------------------------
# Unified hybrid sampler with quantum head
# ----------------------------------------------------------------------
class UnifiedHybridSampler(nn.Module):
    """End‑to‑end sampler that uses a quantum circuit for the final decision."""
    def __init__(self, n_fraud_layers: int = 3, shots: int = 200, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        dummy_params = FraudLayerParameters(
            bs_theta=0.1, bs_phi=0.2,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud = build_fraud_detection_program(dummy_params, [dummy_params]*n_fraud_layers)
        self.hybrid = HybridLayer(shots=shots, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.softmax(self.sampler(x), dim=-1)
        out = self.fraud(out)
        out = self.hybrid(out.squeeze(-1))
        return out

__all__ = ["UnifiedHybridSampler"]
