import numpy as np
import qiskit
from qiskit import assemble, transpile
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumCircuit:
    """Parameterised 4‑qubit quantum circuit executed on Aer."""
    def __init__(self, backend=None, shots: int = 1024) -> None:
        if backend is None:
            backend = qiskit.Aer.get_backend("aer_simulator")
        self.n_qubits = 4
        self.backend = backend
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.params = qiskit.circuit.ParameterVector('theta', self.n_qubits)
        self.circuit.h(range(self.n_qubits))
        self.circuit.ry(self.params, range(self.n_qubits))
        self.circuit.cx(0, 1)
        self.circuit.cx(2, 3)
        self.circuit.measure_all()

    def _expectation(self, counts: dict) -> float:
        total = sum(counts.values())
        exp = 0.0
        for state, cnt in counts.items():
            prob = cnt / total
            value = int(state, 2)
            exp += value * prob
        return exp

    def run(self, thetas: np.ndarray) -> np.ndarray:
        expectations = []
        for theta in thetas:
            bind_dict = {self.params[i]: theta[i] for i in range(self.n_qubits)}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind_dict])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts(self.circuit)
            expectations.append(self._expectation(counts))
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.circuit = circuit
        ctx.shift = shift
        expectations = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        batch, n_qubits = inputs.shape
        grads = []
        for i in range(batch):
            grad_i = []
            for j in range(n_qubits):
                shift_vec = np.zeros(n_qubits)
                shift_vec[j] = shift
                right = ctx.circuit.run((inputs[i] + shift_vec).unsqueeze(0))[0]
                left = ctx.circuit.run((inputs[i] - shift_vec).unsqueeze(0))[0]
                grad_i.append(right - left)
            grads.append(grad_i)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device)
        return grads * grad_output.unsqueeze(-1), None, None

class HybridHeadQuantum(nn.Module):
    """Quantum head that outputs a single expectation value."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(backend=backend, shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

class ConvBackbone(nn.Module):
    """CNN backbone identical to the classical version."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.norm(x)
        return x

class HybridQuantumClassifier(nn.Module):
    """Hybrid quantum‑classical binary classifier with a true quantum head."""
    def __init__(self, n_qubits: int = 4, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.backbone = ConvBackbone()
        self.head = HybridHeadQuantum(n_qubits=n_qubits, backend=backend, shots=shots, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumClassifier"]
