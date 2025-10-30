"""Hybrid quantum‑classical binary classifier with a photonic‑style
parameterised circuit.

This module extends the original `ClassicalQuantumBinaryClassification`
by replacing its simple linear head with a two‑qubit variational circuit
whose parameters are organised using a `QuantumLayerParameters`
dataclass.  The circuit is assembled only once per forward pass,
allowing gradients to flow through Qiskit’s parameter‑binding API via a
custom `HybridFunction`.  Parameters are clipped to emulate the
photonic implementation’s saturation limits.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable
import qiskit
from qiskit import assemble, transpile

@dataclass
class QuantumLayerParameters:
    """Parameters describing a single photonic‑style quantum layer."""
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

class QuantumCircuit:
    """Parameterised two‑qubit circuit with photonic‑style gates."""
    def __init__(self,
                 params: QuantumLayerParameters,
                 backend,
                 shots: int,
                 clip: bool = True) -> None:
        self.backend = backend
        self.shots = shots
        self.clip = clip
        self.params = params
        self._circuit = qiskit.QuantumCircuit(2)
        theta = qiskit.circuit.Parameter("theta")
        self.theta = theta
        # Build base circuit
        self._circuit.h([0, 1])
        self._circuit.barrier()
        self._circuit.ry(theta, 0)
        self._circuit.ry(theta, 1)
        # Fixed photonic‑style gates
        for i, phase in enumerate(params.phases):
            self._circuit.rz(phase, i)
        for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r = _clip(r, 5.0) if self.clip else r
            ph = _clip(ph, np.pi)
            self._circuit.rx(r, i)
            self._circuit.rz(ph, i)
        for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r = _clip(r, 5.0) if self.clip else r
            ph = _clip(ph, np.pi)
            self._circuit.ry(r, i)
            self._circuit.rz(ph, i)
        for i, k in enumerate(params.kerr):
            k = _clip(k, 1.0) if self.clip else k
            self._circuit.rz(k, i)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas]
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.tolist())
        output = torch.tensor(expectation, dtype=torch.float32)
        ctx.save_for_backward(inputs, output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for val in inputs.tolist():
            right = ctx.circuit.run([val + shift[0]])
            left  = ctx.circuit.run([val - shift[0]])
            grads.append(right - left)
        grad_inputs = torch.tensor(grads, dtype=torch.float32) * grad_output
        return grad_inputs, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards a scalar through the quantum circuit."""
    def __init__(self,
                 params: QuantumLayerParameters,
                 backend,
                 shots: int = 1024,
                 shift: float = np.pi / 2,
                 clip: bool = True) -> None:
        super().__init__()
        self.shift = shift
        self.circuit = QuantumCircuit(params, backend, shots, clip=clip)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.circuit, self.shift)

class HybridQCNet(nn.Module):
    """CNN followed by a quantum expectation head."""
    def __init__(self,
                 qc_params: QuantumLayerParameters,
                 backend=None,
                 shots: int = 1024,
                 shift: float = np.pi / 2
                 ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(qc_params, backend, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumLayerParameters", "HybridQCNet"]
