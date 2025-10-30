import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on AerSimulator."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        # Entangle qubits with a CX chain
        for q in range(n_qubits - 1):
            self.circuit.cx(q, q + 1)
        # Apply a common rotation to each qubit
        self.circuit.ry(self.theta, list(range(n_qubits)))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of rotation angles."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Expectation value of Pauli Z on the first qubit
        def expectation(counts):
            probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0)
                              for i in range(2 ** self.n_qubits)]) / self.shots
            z_vals = np.array([(-1) ** int(bin(i)[2:].zfill(self.n_qubits)[0]) for i in range(2 ** self.n_qubits)])
            return np.sum(z_vals * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = np.pi / 2):
        ctx.shift = shift
        ctx.circuit = circuit
        # Run the quantum circuit on CPU to keep it simple
        thetas = inputs.detach().cpu().numpy().flatten()
        expectations = circuit.run(thetas)
        result = torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for x in inputs.detach().cpu().numpy():
            eps = shift
            right = ctx.circuit.run([x + eps])[0]
            left = ctx.circuit.run([x - eps])[0]
            grads.append(right - left)
        grad_inputs = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grad_inputs * grad_output, None, None

class HybridLayer(nn.Module):
    """Layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate states and a sinusoidal target for regression, then binarise."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    # Binarise using a threshold at zero
    labels = (y > 0).astype(np.float32)
    return x, labels

class SuperpositionDataset(torch.utils.data.Dataset):
    """Dataset of superposition states with binary labels."""
    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class HybridBinaryClassifier(nn.Module):
    """CNN + quantum expectation head for binary classification."""
    def __init__(self, in_channels: int = 3, num_features: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(15),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        self.flat_dim = 15 * 7 * 7  # assuming 32x32 input images
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(84, 1),
        )
        self.hybrid = HybridLayer(n_qubits=2, shots=512, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.hybrid(x)
        probs = torch.sigmoid(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier", "SuperpositionDataset", "generate_superposition_data"]
