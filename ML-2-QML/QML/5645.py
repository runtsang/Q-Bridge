from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import transpile, assemble
from qiskit.circuit import Parameter

class QuantumCircuitWrapper:
    """Two‑parameter variational circuit with a single qubit."""
    def __init__(self, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.shift = shift
        self.input_param = Parameter("theta_in")
        self.weight_param = Parameter("theta_wt")
        self.circuit = qiskit.QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)
        self.circuit.measure_all()
        self.observable = qiskit.quantum_info.SparsePauliOp.from_list([("Y", 1)])

    def expectation(self, input_val: float, weight_val: float) -> float:
        bound_circ = self.circuit.bind_parameters({
            self.input_param: input_val,
            self.weight_param: weight_val
        })
        compiled = transpile(bound_circ, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys())).astype(float)
        return float(np.sum(states * probs))

class QuantumHybridFunction(torch.autograd.Function):
    """Autograd wrapper that applies the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        exp_vals = []
        for val in inputs.detach().cpu().numpy():
            exp_vals.append(circuit.expectation(val, circuit.weight.item()))
        return torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grad_inputs = []
        for val in inputs.detach().cpu().numpy():
            exp_plus = circuit.expectation(val + shift, circuit.weight.item())
            exp_minus = circuit.expectation(val - shift, circuit.weight.item())
            grad_inputs.append((exp_plus - exp_minus) / 2.0)
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32, device=inputs.device)

        # weight gradient via finite difference
        w = circuit.weight.item()
        exp_plus_w = circuit.expectation(inputs.detach().cpu().numpy()[0], w + shift)
        exp_minus_w = circuit.expectation(inputs.detach().cpu().numpy()[0], w - shift)
        grad_weight = (exp_plus_w - exp_minus_w) / 2.0
        grad_weight = torch.tensor([grad_weight], dtype=torch.float32, device=inputs.device)

        return grad_inputs * grad_output, None, None

class QuantumHybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(shots=shots, shift=shift)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return QuantumHybridFunction.apply(squeezed, self.circuit, self.shift)

class HybridQCNet(nn.Module):
    """CNN backbone followed by a quantum expectation head."""
    def __init__(self, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = QuantumHybrid(shots=shots, shift=shift)

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
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["HybridQCNet"]
