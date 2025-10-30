import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import transpile, assemble

class QuantumCircuit:
    """Variational circuit with a parameter vector, executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.params = qiskit.circuit.ParameterVector('theta', length=n_qubits)
        # Simple ansatz: RX rotations + CZ entanglement
        for i in range(n_qubits):
            self.circuit.rx(self.params[i], i)
        for i in range(n_qubits - 1):
            self.circuit.cz(i, i + 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameter vectors."""
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        expectations = []
        for theta in thetas:
            bound = self.circuit.bind_parameters({self.params[i]: theta[i] for i in range(self.n_qubits)})
            compiled = transpile(bound, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            expectation = 0.0
            for bitstring, count in counts.items():
                # Map '0' -> +1, '1' -> -1 for the first qubit
                z = 1 if bitstring[0] == '0' else -1
                expectation += z * count
            expectation /= self.shots
            expectations.append(expectation)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit using parameter shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        expectations = circuit.run(thetas)
        result = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        thetas = inputs.detach().cpu().numpy()
        shift = ctx.shift
        grads = []
        for theta in thetas:
            theta_plus = theta + shift
            theta_minus = theta - shift
            exp_plus = ctx.circuit.run(theta_plus.reshape(1, -1))[0]
            exp_minus = ctx.circuit.run(theta_minus.reshape(1, -1))[0]
            grads.append((exp_plus - exp_minus) / 2.0)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output.unsqueeze(1), None, None

class AttentionBlock(nn.Module):
    """Simple attention that learns a weight per feature."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight

class CalibrationModule(nn.Module):
    """Adds a learnable bias to the output."""
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias

class HybridQuantumBinaryClassifier(nn.Module):
    """CNN followed by an attention block and a variational quantum head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.attention = AttentionBlock(84)
        self.proj = nn.Linear(84, 4)  # map to 4 qubit parameters
        backend = qiskit.Aer.get_backend('aer_simulator')
        self.quantum_circuit = QuantumCircuit(4, backend, shots=1024)
        self.shift = np.pi / 2
        self.calibration = CalibrationModule()
        self.fc3 = nn.Linear(1, 1)

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
        x = self.attention(x)
        params = self.proj(x)  # (batch, 4)
        quantum_out = HybridFunction.apply(params, self.quantum_circuit, self.shift)
        quantum_out = quantum_out.unsqueeze(1)  # (batch, 1)
        quantum_out = self.fc3(quantum_out)
        quantum_out = self.calibration(quantum_out)
        probs = torch.sigmoid(quantum_out)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
