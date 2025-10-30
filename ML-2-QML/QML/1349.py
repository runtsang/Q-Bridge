import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuitWrapper:
    """
    A two‑qubit parameterised circuit used as the quantum head.
    The circuit consists of a Hadamard layer, a CNOT, and a single
    Ry rotation on qubit 0.  The expectation value of Z on qubit 0
    is returned as a differentiable output.
    """
    def __init__(self, n_qubits: int = 2, shots: int = 500):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.base_circ = QC(self.n_qubits)
        self.base_circ.h(range(self.n_qubits))
        self.base_circ.cx(0, 1)
        self.theta = QC.Parameter("θ")
        self.base_circ.ry(self.theta, 0)
        self.base_circ.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each parameter in `params` and return
        the expectation value of Z on qubit 0.
        """
        expectations = []
        for val in params:
            circ = self.base_circ.copy()
            circ.assign_parameters({self.theta: val}, inplace=True)
            compiled = transpile(circ, self.backend, basis_gates=["u3", "cx"])
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, cnt in counts.items():
                z = 1 if bitstring[0] == "0" else -1
                exp += z * cnt
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations, dtype=np.float32)

class QuantumExpectationFunction(torch.autograd.Function):
    """
    Autograd wrapper that evaluates the quantum circuit and applies
    the parameter‑shift rule for gradients.
    """
    @staticmethod
    def forward(ctx, params: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.save_for_backward(params)
        outputs = circuit.run(params.detach().cpu().numpy())
        return torch.from_numpy(outputs).to(params.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        params, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        plus = params + shift
        minus = params - shift
        out_plus = circuit.run(plus.detach().cpu().numpy())
        out_minus = circuit.run(minus.detach().cpu().numpy())
        grad = (out_plus - out_minus) / (2 * shift)
        grad_tensor = torch.from_numpy(grad).to(params.device)
        return grad_tensor * grad_output, None, None

class HybridQuantumBinaryClassifier(nn.Module):
    """
    CNN backbone followed by a quantum expectation head.  The quantum
    circuit is transpiled to a custom basis‑gate set and each sample
    is evaluated with a per‑sample shot budget.  The head can be
    switched to a classical surrogate by setting `use_classical=True`.
    """
    def __init__(self, shots_per_sample: int = 200, shift: float = np.pi / 2, use_classical: bool = False):
        super().__init__()
        self.shift = shift
        self.use_classical = use_classical
        # Backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        # Quantum head
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits=2, shots=shots_per_sample)
        self.quantum_head = QuantumExpectationFunction.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x).squeeze(-1)
        if self.use_classical:
            probs = torch.sigmoid(logits + self.shift)
        else:
            shifted = logits + self.shift
            probs = self.quantum_head(shifted, self.quantum_circuit, self.shift)
            probs = torch.sigmoid(probs)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
