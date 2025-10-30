import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile

class QuantumCircuit:
    """Parameterised 4‑qubit hardware‑efficient ansatz."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = [qiskit.circuit.Parameter(f'theta_{i}') for i in range(n_qubits * 2)]
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        # First layer of RY rotations
        for i in range(n_qubits):
            self._circuit.ry(self.theta[i], i)
        # Entanglement with CX gates
        for i in range(n_qubits - 1):
            self._circuit.cx(i, i + 1)
        # Second layer of RY rotations
        for i in range(n_qubits):
            self._circuit.ry(self.theta[n_qubits + i], i)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of theta vectors.
        Returns the expectation value of Z on the first qubit for each batch."""
        expectations = []
        for theta_vec in thetas:
            param_bind = {self.theta[i]: val for i, val in enumerate(theta_vec)}
            compiled = transpile(self._circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_bind])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            # Compute expectation of Z on qubit 0
            exp = 0.0
            for bitstring, count in result.items():
                # bitstring ends with qubit 0
                z = 1.0 if bitstring[-1] == '0' else -1.0
                exp += z * count / self.shots
            expectations.append(exp)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Custom autograd function that forwards through a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        batch_thetas = inputs.detach().cpu().numpy()
        expectations = circuit.run(batch_thetas)
        return torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output):
        shift = ctx.shift
        circuit = ctx.circuit
        inputs = grad_output.detach().cpu().numpy()
        grads = []
        for theta_vec in inputs:
            grad = []
            for i, val in enumerate(theta_vec):
                theta_plus = theta_vec.copy()
                theta_minus = theta_vec.copy()
                theta_plus[i] += shift
                theta_minus[i] -= shift
                exp_plus = circuit.run([theta_plus])[0]
                exp_minus = circuit.run([theta_minus])[0]
                grad.append((exp_plus - exp_minus) / (2 * np.sin(shift)))
            grads.append(grad)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output.unsqueeze(-1), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridClassifier(nn.Module):
    """CNN feature extractor followed by a 4‑qubit quantum expectation head."""
    def __init__(self):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully‑connected head that outputs 4 parameters for the 4‑qubit circuit
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)
        # Quantum hybrid head
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=4, backend=backend, shots=200, shift=np.pi / 2)

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
        # Forward through the quantum circuit
        probs = self.hybrid(x)
        probs = torch.sigmoid(probs)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridClassifier"]
