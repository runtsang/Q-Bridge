import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator
from typing import Sequence

class QuantumCircuit:
    """
    Parametrised two‑qubit circuit with a single rotation parameter.
    The circuit is executed on the Aer simulator and returns the
    expectation value of the Z‑observable on the first qubit.
    """

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        self.theta = qiskit.circuit.Parameter("theta")
        self.circ = qiskit.QuantumCircuit(n_qubits)
        self.circ.h(range(n_qubits))
        self.circ.cx(0, 1)
        self.circ.ry(self.theta, range(n_qubits))
        self.circ.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each value in *thetas* and return the
        corresponding expectation values.
        """
        expectations = []
        for theta in thetas:
            bound_circ = transpile(self.circ, self.backend)
            bound_circ = bound_circ.bind_parameters({self.theta: theta})
            qobj = assemble(bound_circ, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            # Expectation of Z on the first qubit
            exp = 0.0
            for bitstring, count in result.items():
                bit = int(bitstring[0])  # first qubit
                exp += ((-1) ** bit) * count
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards a batch of activations through
    the quantum circuit and implements a finite‑difference gradient.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        batch = inputs.cpu().numpy().flatten()
        exp_vals = circuit.run(batch)
        result = torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in inputs.cpu().numpy().flatten():
            right = circuit.run([val + shift])[0]
            left = circuit.run([val - shift])[0]
            grads.append(right - left)
        grad_inputs = torch.tensor(grads, dtype=torch.float32, device=inputs.device).unsqueeze(-1) * grad_output
        return grad_inputs, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridQuantumClassifier(nn.Module):
    """
    Convolutional network followed by a variational quantum expectation head.
    The architecture is an extension of the original hybrid model:
    - A configurable dropout schedule for the convolutional layers.
    - A small MLP head before the quantum layer.
    - A two‑qubit variational circuit with an entangling gate.
    """

    def __init__(
        self,
        conv_dropout: Sequence[float] = (0.2, 0.5),
        mlp_dropout: Sequence[float] = (0.3, 0.4),
        mlp_hidden: int = 120,
        mlp_second_hidden: int = 84,
        n_qubits: int = 2,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=conv_dropout[0])
        self.drop2 = nn.Dropout2d(p=conv_dropout[1])

        dummy = torch.zeros(1, 3, 32, 32)
        dummy = self._forward_conv(dummy)
        flat_features = dummy.size(1)

        self.fc1 = nn.Linear(flat_features, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_second_hidden)
        self.fc3 = nn.Linear(mlp_second_hidden, 1)

        self.dropout1 = nn.Dropout(p=mlp_dropout[0])
        self.dropout2 = nn.Dropout(p=mlp_dropout[1])

        self.hybrid = Hybrid(
            n_qubits=n_qubits,
            backend=AerSimulator(),
            shots=shots,
            shift=shift,
        )

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop2(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x).squeeze(-1)
        logits = logits.unsqueeze(-1)
        probs = self.hybrid(logits).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)
