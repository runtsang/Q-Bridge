import torch
import torch.nn as nn
import numpy as np
import qiskit
from qiskit import Aer, assemble, transpile

class QuantumCircuit:
    """
    A parametrised two‑qubit circuit executed on the Aer simulator.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        if backend is None:
            backend = Aer.get_backend('aer_simulator')
        self.backend = backend
        self.shots = shots
        self.n_qubits = n_qubits
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            total = sum(counts.values())
            exp_val = 0.0
            for state, cnt in counts.items():
                bit = int(state[0])
                exp_val += (1 - 2 * bit) * cnt
            return exp_val / total
        return np.array([expectation(result)])

    def run_batch(self, thetas: np.ndarray) -> np.ndarray:
        return np.concatenate([self.run([t]) for t in thetas], axis=0)

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = circuit.run_batch(inputs.cpu().numpy())
        result = torch.from_numpy(expectations).to(inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.cpu().numpy(), ctx.shift)
        grads = []
        for idx, val in enumerate(inputs.cpu().numpy()):
            right = ctx.circuit.run([val + shift[idx]])
            left = ctx.circuit.run([val - shift[idx]])
            grads.append((right - left) / 2.0)
        grads = torch.from_numpy(np.array(grads)).to(inputs.device)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x.squeeze(), self.circuit, self.shift)

class HybridQuantumBinaryClassifier(nn.Module):
    """
    Hybrid quantum‑classical binary classifier.
    Feature extractor identical to the classical baseline,
    but the final head uses a parameterised quantum circuit.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten()
        )
        dummy = torch.zeros(1, 3, 32, 32)
        out = self.feature_extractor(dummy)
        in_features = out.shape[1]
        self.fc = nn.Linear(in_features, n_qubits)
        if backend is None:
            backend = Aer.get_backend('aer_simulator')
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.fc(features)
        probs = self.hybrid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)
