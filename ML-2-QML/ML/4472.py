import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional
import numpy as np
import qiskit

# --------------------------------------------------------------------------- #
#  Classical fraud‑detection backbone (photonic‑style linear layers)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic‑style linear block."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _make_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a linear block that mirrors the photonic layer but uses a
    standard PyTorch linear + tanh activation."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            y = y * self.scale + self.shift
            return y

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a sequential model that implements the classical backbone."""
    modules: list[nn.Module] = [_make_layer(input_params, clip=False)]
    modules.extend(_make_layer(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
#  Quantum‑enhanced head (two‑qubit Qiskit circuit)
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """A parameterised two‑qubit circuit that returns the expectation of Z."""
    def __init__(self, backend, shots: int = 200, seed: int | None = None):
        self._circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple entangling pattern: H‑R_y(θ)‑CNOT‑R_y(θ)‑CNOT
        self._circuit.h([0, 1])
        self._circuit.barrier()
        self._circuit.ry(self.theta, [0, 1])
        self._circuit.cx([0, 1])
        self._circuit.ry(self.theta, [0, 1])
        self._circuit.cx([0, 1])
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.seed = seed

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """Execute the circuit for each θ in *thetas* and return a tensor."""
        compiled = qiskit.transpile(self._circuit, self.backend)
        qobj = qiskit.assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts(self._circuit)
        # Convert counts to expectation value of Z
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()), dtype=float)
            return np.sum(states * probs)
        if isinstance(result, list):
            return torch.tensor([expectation(r) for r in result], dtype=torch.float32)
        return torch.tensor([expectation(result)], dtype=torch.float32)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        exp = circuit.run(inputs.tolist())
        out = torch.tensor(exp, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.numpy(), ctx.shift)
        gradients = []
        for v in inputs.numpy():
            g = ctx.circuit.run([v + shift])
            h = ctx.circuit.run([v - shift])
            gradients.append(g - h)
        grad = torch.tensor(gradients, dtype=torch.float32)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum = QuantumCircuit(backend, shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x.squeeze(), self.quantum, self.shift)

class QCNet(nn.Module):
    """Full hybrid model combining the classical backbone and a quantum head."""
    def __init__(self, backbone: nn.Sequential, n_qubits: int = 2, backend=None, shots=200, shift=np.pi/2):
        super().__init__()
        self.backbone = backbone
        self.hybrid = Hybrid(n_qubits, backend or qiskit.Aer.get_backend("qasm_simulator"), shots=shots, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        out = self.hybrid(out)
        return torch.cat((out, 1 - out), dim=-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuantumCircuit",
           "HybridFunction", "Hybrid", "QCNet"]
