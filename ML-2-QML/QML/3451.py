import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class QuantumExpectation:
    """Variational circuit with parameter‑shift gradient, returning expectation of Z on qubit 0."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024, backend=None):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.theta = qiskit.circuit.ParameterVector('theta', n_qubits)
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        # Random layer: H on all qubits
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        # Parameterised Ry gates
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        # Entangling layer
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for each row of thetas."""
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        exps = []
        for row in thetas:
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled,
                            shots=self.shots,
                            parameter_binds=[{self.theta[i]: th for i, th in enumerate(row)}])
            job = self.backend.run(qobj)
            counts = job.result().get_counts()
            exp = 0.0
            for state, cnt in counts.items():
                # qubit 0 is the last bit in the state string
                z = 1 if state[-1] == '0' else -1
                exp += z * cnt
            exps.append(exp / self.shots)
        return np.array(exps)

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper around QuantumExpectation using the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumExpectation, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        exp_vals = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run([val + shift])[0]
            left = ctx.circuit.run([val - shift])[0]
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=inputs.dtype, device=inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum layer mapping 4‑dimensional features to a single expectation."""
    def __init__(self, n_qubits: int = 4, shift: float = np.pi / 2, shots: int = 1024):
        super().__init__()
        self.circuit = QuantumExpectation(n_qubits=n_qubits, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, 4)
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """CNN + 4‑feature projection + quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        self.fc_proj = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        self.hybrid = Hybrid(n_qubits=4, shift=np.pi / 2, shots=1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc_proj(x)
        x = self.norm(x)
        probs = self.hybrid(x).squeeze(-1)
        probs = torch.sigmoid(probs)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier"]
