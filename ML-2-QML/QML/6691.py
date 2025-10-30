import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator

class VariationalCircuit:
    """A multi‑qubit variational circuit with learnable rotation axes."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.theta = [QuantumCircuit.Parameter(f"theta_{i}") for i in range(n_qubits)]
        self.circuit = QuantumCircuit(n_qubits)
        # Simple layered ansatz
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit with the provided parameters."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{p: val} for p, val in zip(self.theta, params)],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        # Expectation value of Z on qubit 0
        exp = 0.0
        for state, count in counts.items():
            z = 1 if state[-1] == '0' else -1
            exp += z * count
        exp /= self.shots
        return np.array([exp])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface that forwards activations through the variational circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Execute for each sample in the batch
        exp_vals = []
        for inp in inputs.detach().cpu().numpy():
            exp_vals.append(ctx.circuit.run(inp))
        result = torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for val in inputs.detach().cpu().numpy():
            right = ctx.circuit.run(np.array([val + shift]))
            left = ctx.circuit.run(np.array([val - shift]))
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output, None, None

class HybridNet(nn.Module):
    """Hybrid CNN with a quantum variational head."""
    def __init__(self, n_qubits: int = 4, shift: float = np.pi/2, shots: int = 200):
        super().__init__()
        # Convolutional backbone (identical to the classical version)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Determine feature size after conv layers
        dummy = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            out = self._extract_features(dummy)
        in_features = out.shape[1]
        # Quantum variational head
        backend = AerSimulator()
        self.quantum_head = VariationalCircuit(n_qubits, backend, shots)
        self.shift = shift

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(x)
        # Map features to a vector of length n_qubits per sample
        param = torch.mean(features, dim=1, keepdim=True)
        param = param.repeat(1, self.quantum_head.n_qubits)
        q_out = HybridFunction.apply(param, self.quantum_head, self.shift)
        probs = torch.sigmoid(q_out)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridNet"]
