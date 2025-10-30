import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuantumCircuitWrapper:
    """Quantum circuit executing parametrised expectation on Aer simulator."""
    def __init__(self, n_qubits, backend=None, shots=100):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between torch tensors and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grads = []
        for val, s in zip(inputs.cpu().numpy(), shift):
            right = ctx.circuit.run([val + s])
            left = ctx.circuit.run([val - s])
            grads.append(right - left)
        grads = torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device)
        return grads * grad_output, None, None

class HybridHead(nn.Module):
    """Quantum expectation head used inside the hybrid classifier."""
    def __init__(self, n_qubits: int, shift: float = np.pi/2, backend=None, shots: int = 100):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend=backend, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift).unsqueeze(-1)

class UnifiedQuantumClassifier(nn.Module):
    """
    Hybrid classifier that can operate in three modes:
    1. Pure classical MLP.
    2. Hybrid MLP + quantum head.
    3. Pure quantum (if classical backbone is a single linear).
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 3,
                 use_quantum: bool = True,
                 n_qubits: int = 2,
                 shift: float = np.pi/2,
                 backend=None,
                 shots: int = 100):
        super().__init__()
        self.use_quantum = use_quantum
        layers = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            in_dim = num_features
        self.backbone = nn.Sequential(*layers)
        self.fc_out = nn.Linear(num_features, 1)
        if use_quantum:
            self.head = HybridHead(n_qubits, shift=shift, backend=backend, shots=shots)
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        logits = self.fc_out(x)
        probs = self.head(logits.squeeze(-1))
        probs = torch.sigmoid(probs)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["UnifiedQuantumClassifier", "HybridFunction", "HybridHead", "QuantumCircuitWrapper"]
