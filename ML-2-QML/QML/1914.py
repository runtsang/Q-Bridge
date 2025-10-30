from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumCircuit:
    """
    Parameterised two‑qubit circuit with entanglement.
    The circuit consists of:
        • Global Hadamards
        • Parametric RY rotations
        • CX entanglement
        • Measurement in the computational basis
    """
    def __init__(self, n_qubits: int, backend: qiskit.providers.Provider, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend   = backend
        self.shots     = shots

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.ParameterVector("theta", n_qubits)

        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        # Entangle all qubits
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for the supplied angle vector(s) and return
        the expectation value of Z on the first qubit.
        """
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job  = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts: dict) -> float:
            total = sum(counts.values())
            exp = 0.0
            for state, cnt in counts.items():
                # Interpret bitstring as integer, Z eigenvalue = 1-2*bit
                bit = int(state[0])  # first qubit
                exp += (1 - 2 * bit) * (cnt / total)
            return exp

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    PyTorch autograd wrapper that forwards data through the quantum
    circuit and implements the parameter‑shift gradient.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = circuit.run(inputs.cpu().numpy().flatten())
        out = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.pi / 2
        grads = []
        for val in inputs.cpu().numpy().flatten():
            exp_plus  = ctx.circuit.run([val + shift])[0]
            exp_minus = ctx.circuit.run([val - shift])[0]
            grads.append((exp_plus - exp_minus) / 2.0)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output, None, None


class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """
    def __init__(self, n_qubits: int, backend: qiskit.providers.Provider,
                 shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Expectation value returned as a tensor of shape (batch, 1)
        return HybridFunction.apply(inputs.squeeze(-1), self.circuit, self.shift).unsqueeze(-1)


class QCNet(nn.Module):
    """
    Hybrid CNN + quantum expectation head for binary classification.
    Architecture mirrors the classical QCNet but replaces the final
    sigmoid head with a quantum‑parameterised layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool  = nn.MaxPool2d(2)
        self.drop  = nn.Dropout2d(p=0.5)
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1   = nn.Linear(128 * 4 * 4, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 1)

        aer = AerSimulator()
        self.hybrid = Hybrid(
            n_qubits=1,
            backend=aer,
            shots=1024,
            shift=np.pi / 2
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
