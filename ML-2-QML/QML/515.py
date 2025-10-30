import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return x + self.bn(self.conv(x))

class QuantumCircuitWrapper:
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.add_parameter("theta")
        self.circuit.h(0)
        self.circuit.ry("theta", 0)
        self.circuit.measure_all()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def expectation(self, theta: float) -> float:
        bound = {self.circuit.params[0]: theta}
        transpiled = qiskit.transpile(self.circuit, self.backend)
        job = qiskit.execute(transpiled, backend=self.backend, shots=self.shots,
                             parameter_binds=[bound])
        result = job.result()
        counts = result.get_counts()
        p0 = sum(counts[bit] for bit in counts if bit[-1] == '0') / self.shots
        p1 = 1 - p0
        expectation = p0 - p1
        return expectation

class QuantumExpectationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, circuit, shots):
        expectation = circuit.expectation(theta.item())
        out = torch.tensor([expectation], dtype=torch.float32)
        ctx.save_for_backward(theta)
        ctx.circuit = circuit
        ctx.shots = shots
        return out

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        shift = np.pi / 2
        e_plus = ctx.circuit.expectation(theta.item() + shift)
        e_minus = ctx.circuit.expectation(theta.item() - shift)
        grad = (e_plus - e_minus) / 2
        return grad_output * grad, None, None

class QuantumHybridBinaryClassifier(nn.Module):
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.res = ResidualBlock(15)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum = QuantumCircuitWrapper(n_qubits, shots)
        self.shots = shots

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.res(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        theta = x.squeeze()
        quantum_out = QuantumExpectationFunction.apply(theta, self.quantum, self.shots)
        prob = torch.sigmoid(quantum_out)
        return torch.cat((prob, 1 - prob), dim=-1)
