import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit as QiskitCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class ParametrizedTwoQubitCircuit:
    """Two‑qubit circuit: H on both, RY(theta) on each, CX, Z measurement on qubit 0."""
    def __init__(self, backend: AerSimulator, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circuit = QiskitCircuit(2)
        self.theta = QiskitCircuit.Parameter('theta')
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def expectation(self, params: np.ndarray) -> np.ndarray:
        """Return expectation value of Z on qubit 0 for each parameter."""
        exp = []
        for p in params:
            binding = {self.theta: float(p)}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[binding])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp_val = 0.0
            for bitstring, cnt in counts.items():
                z = 1 if bitstring[0] == '0' else -1
                exp_val += z * cnt
            exp_val /= self.shots
            exp.append(exp_val)
        return np.array(exp)

class ParameterShiftFunction(torch.autograd.Function):
    """Compute expectation of a two‑qubit circuit with parameter‑shift gradient."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: ParametrizedTwoQubitCircuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        with torch.no_grad():
            exp = circuit.expectation(inputs.cpu().numpy())
        result = torch.tensor(exp, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        with torch.no_grad():
            exp_plus = ctx.circuit.expectation((inputs + shift).cpu().numpy())
            exp_minus = ctx.circuit.expectation((inputs - shift).cpu().numpy())
        grad_inputs = (exp_plus - exp_minus) / (2 * shift)
        grad_inputs = torch.tensor(grad_inputs, dtype=grad_output.dtype, device=grad_output.device)
        return grad_inputs * grad_output, None, None

class HybridQuantumLayer(nn.Module):
    """Hybrid layer mapping a scalar to a probability via a two‑qubit circuit."""
    def __init__(self, backend: AerSimulator, shots: int = 512, shift: float = np.pi/2):
        super().__init__()
        self.shift = shift
        self.circuit = ParametrizedTwoQubitCircuit(backend, shots)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        exp = ParameterShiftFunction.apply(inputs, self.circuit, self.shift)
        out = torch.sigmoid(exp + self.bias)
        return out

class HybridClassifier(nn.Module):
    """CNN followed by a quantum expectation head producing two class probabilities."""
    def __init__(self) -> None:
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
        self.hybrid = HybridQuantumLayer(backend, shots=512, shift=np.pi/2)

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
        x = self.fc3(x).squeeze(-1)
        prob = self.hybrid(x)
        return torch.cat((prob.unsqueeze(-1), (1 - prob).unsqueeze(-1)), dim=-1)

__all__ = ['HybridClassifier', 'HybridQuantumLayer', 'ParameterShiftFunction', 'ParametrizedTwoQubitCircuit']
