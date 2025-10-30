import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import Aer, assemble, transpile
from qiskit.circuit import Parameter

class VariationalAnsatz:
    """A two‑qubit variational ansatz with tunable entanglement."""
    def __init__(self, n_qubits: int = 2, entanglement: str = "cnot"):
        self.n_qubits = n_qubits
        self.entanglement = entanglement
        self.theta = Parameter("θ")

    def build(self, circuit: qiskit.QuantumCircuit):
        # Apply a rotation to each qubit
        for q in range(self.n_qubits):
            circuit.ry(self.theta, q)
        # Add entangling gate
        if self.entanglement == "cnot":
            circuit.cx(0, 1)
        elif self.entanglement == "cz":
            circuit.cz(0, 1)

class QuantumCircuit:
    """Wrapper around a parametrised circuit that can be executed on a backend."""
    def __init__(self, n_qubits: int, backend, shots: int, ansatz: VariationalAnsatz):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.ansatz = ansatz
        self.ansatz.build(self.circuit)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.ansatz.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            total = sum(counts.values())
            exp_val = 0.0
            for state, cnt in counts.items():
                exp_val += int(state, 2) * cnt
            return exp_val / total

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            exp_r = ctx.quantum_circuit.run([value + shift[idx]])
            exp_l = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(exp_r - exp_l)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid head that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float, ansatz: VariationalAnsatz):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots, ansatz)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class QCNet(nn.Module):
    """CNN backbone followed by a variational quantum hybrid head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = Aer.get_backend("aer_simulator")
        ansatz = VariationalAnsatz(n_qubits=1, entanglement="cnot")
        self.hybrid = Hybrid(
            n_qubits=1,
            backend=backend,
            shots=200,
            shift=np.pi / 2,
            ansatz=ansatz,
        )

    def forward(self, inputs: torch.Tensor):
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

__all__ = ["VariationalAnsatz", "QuantumCircuit", "HybridFunction", "Hybrid", "QCNet"]
