"""HybridQuantumNet: Classical backbone with a quantum circuit head.

This module mirrors the ML version but replaces the quantum kernel
with a full two‑qubit variational circuit executed on Qiskit’s Aer
simulator.  The circuit is wrapped in a PyTorch autograd function
so gradients can be back‑propagated through the quantum expectation
value.  The quantum head is enriched with a learnable shift and
finite‑difference gradient estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple ansatz: H on each qubit, then ry(theta) on each
        for q in range(n_qubits):
            self.circuit.h(q)
            self.circuit.ry(self.theta, q)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Compute expectation of Z on first qubit
        exp = 0.0
        for state, count in result.items():
            prob = count / self.shots
            z = 1 if state[-1] == "0" else -1
            exp += z * prob
        return np.array([exp])

class QuantumLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        outputs = circuit.run(thetas)
        outputs = torch.tensor(outputs, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, outputs = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for i in range(inputs.shape[1]):
            plus = inputs.clone()
            minus = inputs.clone()
            plus[:, i] += shift
            minus[:, i] -= shift
            out_plus = ctx.circuit.run(plus.detach().cpu().numpy())
            out_minus = ctx.circuit.run(minus.detach().cpu().numpy())
            grads.append((out_plus - out_minus) / (2 * shift))
        grad_inputs = torch.tensor(np.stack(grads, axis=1), dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output.unsqueeze(1), None, None

class HybridQuantumNet(nn.Module):
    """CNN backbone with a quantum circuit head."""
    def __init__(self):
        super().__init__()
        # Classical backbone identical to original
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum circuit head
        self.quantum = QuantumCircuitWrapper(n_qubits=2, shots=1024)
        self.shift = np.pi / 2
        self.bn = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
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
        q_out = QuantumLayerFunction.apply(x, self.quantum, self.shift)
        q_out = self.bn(q_out)
        q_out = self.dropout(q_out)
        probs = torch.sigmoid(q_out)
        return torch.cat([probs, 1 - probs], dim=-1)
