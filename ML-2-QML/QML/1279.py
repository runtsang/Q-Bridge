import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class VariationalQuantumCircuit:
    """Parameterized ansatz with 3 qubits and entangling layers."""
    def __init__(self, n_qubits, backend, shots):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter('theta')
        self._circuit = QuantumCircuit(n_qubits)
        self._build_ansatz()

    def _build_ansatz(self):
        # Apply H to all qubits
        self._circuit.h(range(self.n_qubits))
        # Entangling layer
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)
        # Parameterized Ry rotations
        for i in range(self.n_qubits):
            self._circuit.ry(self.theta, i)
        # Measure all qubits
        self._circuit.measure_all()

    def run(self, thetas):
        """Execute the circuit for a batch of theta values."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Compute expectation of Z on qubit 0
        def expectation(count_dict):
            exp = 0.0
            for state, cnt in count_dict.items():
                # state string like '010'
                z = 1 if state[0] == '0' else -1
                exp += z * cnt
            return exp / self.shots
        if isinstance(counts, list):
            return np.array([expectation(c) for c in counts])
        return np.array([expectation(counts)])

class HybridFunction(torch.autograd.Function):
    """Autograd bridge to the variational quantum circuit."""
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        # Run quantum circuit
        exp_vals = ctx.circuit.run(inputs.tolist())
        out = torch.tensor(exp_vals, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in inputs.detach().cpu().numpy():
            exp_plus = circuit.run([val + shift])[0]
            exp_minus = circuit.run([val - shift])[0]
            grads.append(exp_plus - exp_minus)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the variational circuit."""
    def __init__(self, n_qubits, backend, shots, shift):
        super().__init__()
        self.circuit = VariationalQuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x):
        # Ensure shape (batch,)
        flat = x.view(-1)
        return HybridFunction.apply(flat, self.circuit, self.shift)

class QuantumHybridClassifier(nn.Module):
    """CNN followed by a variational quantum expectation head."""
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
        backend = AerSimulator()
        self.hybrid = Hybrid(n_qubits=3, backend=backend, shots=512, shift=np.pi / 2)

    def forward(self, x):
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
        # Quantum hybrid head
        x = self.hybrid(x).view(-1, 1)
        probs = torch.sigmoid(x)
        return torch.cat((probs, 1 - probs), dim=-1)
