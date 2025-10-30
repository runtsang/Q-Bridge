import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """Two‑qubit variational circuit used as the hybrid head."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 200):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        # Simple entangling block
        self._circuit.h(range(self.n_qubits))
        self._circuit.ry(self.theta, 0)
        self._circuit.cx(0, 1)
        self._circuit.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        """Execute the parameterised circuit for each angle in *angles*."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in angles],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            # Expectation of Z⊗I measurement (first qubit)
            probs = {k: v / self.shots for k, v in counts.items()}
            exp = 0.0
            for state, p in probs.items():
                # state string like '00' or '01', first qubit is leftmost
                exp += ((-1) ** int(state[0])) * p
            return exp

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Autograd wrapper that forwards activations through the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to numpy and run
        angles = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(angles)
        ctx.save_for_backward(inputs)
        return torch.tensor(exp_vals, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = np.full_like(inputs.cpu().numpy(), ctx.shift)
        grads = []
        for val in inputs.cpu().numpy():
            right = ctx.circuit.run([val + shift]) - ctx.circuit.run([val - shift])
            grads.append(right)
        grads = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that maps a scalar to a quantum expectation value."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 200, shift: float = np.pi / 4):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class QuantumHybridClassifier(nn.Module):
    """CNN backbone followed by a quantum expectation head."""
    def __init__(self):
        super().__init__()
        # Same feature extractor as the seed model
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Hybrid head
        self.hybrid = Hybrid(n_qubits=2, backend=AerSimulator(), shots=200, shift=np.pi / 4)

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
        x = self.fc3(x).squeeze(-1)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    @staticmethod
    def focal_loss(preds: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
        """Compute focal loss for binary classification."""
        probs = preds[:, 0]
        targets = targets.float()
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = -((1 - pt) ** gamma) * torch.log(pt + eps)
        return loss.mean()

__all__ = ["QuantumHybridClassifier"]
