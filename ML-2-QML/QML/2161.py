import qiskit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

class QuantumCircuit(nn.Module):
    """
    Parameterised two‑qubit variational circuit executed on the Aer simulator.
    The circuit is built once and reused for all forward passes.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 200):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.theta = qiskit.circuit.ParameterVector("theta", 2)
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta[0], 0)
        self.circuit.ry(self.theta[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        expectations = []
        compiled = qiskit.transpile(self.circuit, self.backend)
        for angle in angles:
            bind = {self.theta[0]: angle[0], self.theta[1]: angle[1]}
            qobj = qiskit.assemble(compiled, shots=self.shots, parameter_binds=[bind])
            job = self.backend.run(qobj)
            counts = job.result().get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(s, 2) for s in counts.keys()])
            expectations.append(np.dot(states, probs))
        return np.array(expectations)

class QuantumExpectationFunction(autograd.Function):
    """
    Custom autograd function that forwards the expectation value of a
    parameterised quantum circuit and provides a finite‑difference
    gradient during back‑propagation.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = np.pi/2):
        ctx.shift = shift
        ctx.circuit = circuit
        with torch.no_grad():
            angles = inputs.detach().cpu().numpy()
            expectations = circuit.run(angles)
        ctx.angles = angles
        return torch.tensor(expectations, device=inputs.device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        angles = ctx.angles
        batch_size = angles.shape[0]
        grad_input = np.zeros((batch_size, 2), dtype=np.float32)
        for i in range(2):
            angles_plus = angles.copy()
            angles_minus = angles.copy()
            angles_plus[:, i] += shift
            angles_minus[:, i] -= shift
            E_plus = circuit.run(angles_plus)
            E_minus = circuit.run(angles_minus)
            grad_input[:, i] = (E_plus - E_minus) / (2.0 * shift)
        grad_input = torch.tensor(grad_input, device=grad_output.device, dtype=torch.float32)
        return grad_input * grad_output.unsqueeze(-1), None, None

class QuantumHybridLayer(nn.Module):
    """
    Wraps the QuantumCircuit and exposes a differentiable forward pass
    through the custom autograd function.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 200, shift: float = np.pi/2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumExpectationFunction.apply(x, self.circuit, self.shift)

class HybridNet(nn.Module):
    """
    The hybrid CNN‑quantum network.  The classical backbone processes the image,
    and the final linear layer outputs two parameters for the quantum circuit.
    The quantum expectation is turned into a binary probability.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # produce two parameters for QC
        self.quantum = QuantumHybridLayer()

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
        params = self.fc3(x)  # (batch, 2)
        expectation = self.quantum(params)  # (batch,)
        probs = torch.stack([expectation, 1 - expectation], dim=-1)
        return probs

__all__ = ["HybridNet"]
