import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble

class QuantumCircuitWrapper:
    """Simple two‑qubit circuit with a single variational parameter."""
    def __init__(self, n_qubits: int = 2, shots: int = 200):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Run the circuit for a list of angles."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: float(p)} for p in params],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation of Z on the first qubit
        exp = 0.0
        for state, count in result.items():
            prob = count / self.shots
            bit = int(state[0])
            exp += (-1) ** bit * prob
        return np.array([exp])

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards a batch of logits through a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        exp = torch.from_numpy(circuit.run(inputs.cpu().numpy())).float()
        ctx.save_for_backward(inputs, exp)
        return exp

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in inputs.detach().cpu().numpy():
            exp_plus = circuit.run([val + shift])[0]
            exp_minus = circuit.run([val - shift])[0]
            grads.append((exp_plus - exp_minus) / (2 * shift))
        grads = torch.tensor(grads).float()
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid head that maps a linear projection into a quantum expectation."""
    def __init__(self, in_features: int, n_qubits: int = 2,
                 shots: int = 200, shift: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, n_qubits)
        self.circuit = QuantumCircuitWrapper(n_qubits=n_qubits, shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.linear(x)
        return HybridFunction.apply(params, self.circuit, self.shift)

class ClassifierHead(nn.Module):
    """Auxiliary classical MLP head for uncertainty estimation."""
    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class QCNet(nn.Module):
    """CNN backbone followed by either a quantum or classical head."""
    def __init__(self, use_hybrid: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.use_hybrid = use_hybrid
        self.quantum_head = Hybrid(
            in_features=self.fc3.out_features,
            n_qubits=2,
            shots=200,
            shift=0.5,
        )
        self.classical_head = ClassifierHead(in_features=self.fc3.out_features)

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
        if self.use_hybrid:
            probs = self.quantum_head(x)
        else:
            probs = self.classical_head(x)
        probs = torch.sigmoid(probs)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid",
           "ClassifierHead", "QCNet"]
