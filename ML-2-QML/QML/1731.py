import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit as QiskitCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator

class VariationalQuantumCircuit:
    """Parameterised variational circuit with tunable depth."""
    def __init__(self, n_qubits: int, depth: int, backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots
        self.params = ParameterVector('theta', depth * n_qubits)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = QiskitCircuit(self.n_qubits)
        param_idx = 0
        for _ in range(self.depth):
            # Apply Ry with a unique parameter to each qubit
            for q in range(self.n_qubits):
                qc.ry(self.params[param_idx], q)
                param_idx += 1
            # Entangle adjacent qubits with CX
            for q in range(self.n_qubits - 1):
                qc.cx(q, q + 1)
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray):
        """Evaluate the circuit for a batch of parameter vectors."""
        if thetas.ndim == 1:
            thetas = thetas[np.newaxis, :]
        batch_size = thetas.shape[0]
        param_binds = [{self.params[i]: theta[i] for i in range(len(self.params))}
                       for theta in thetas]
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, self.backend,
                        shots=self.shots,
                        parameter_binds=param_binds)
        job = self.backend.run(qobj)
        results = job.result().get_counts()
        def expectation(counts):
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            z_values = 1 - 2 * ((states >> (self.n_qubits - 1)) & 1)
            return np.sum(z_values * probs)
        if isinstance(results, list):
            return np.array([expectation(r) for r in results])
        else:
            return np.array([expectation(results)])

class QuantumHybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the variational quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalQuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy()
        exp = circuit.run(thetas)
        out = torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_input = torch.zeros_like(inputs)
        batch, dim = inputs.shape
        for i in range(dim):
            theta_plus  = inputs.detach().cpu().numpy()
            theta_minus = inputs.detach().cpu().numpy()
            theta_plus[:, i]  += shift
            theta_minus[:, i] -= shift
            exp_plus  = ctx.circuit.run(theta_plus)
            exp_minus = ctx.circuit.run(theta_minus)
            grad_input[:, i] = torch.tensor(exp_plus - exp_minus, device=inputs.device)
        return grad_input * grad_output, None, None

class QuantumHybridLayer(nn.Module):
    """Quantum expectation layer wrapping the variational circuit."""
    def __init__(self, n_qubits: int, depth: int, shift: float = np.pi / 2):
        super().__init__()
        backend = AerSimulator()
        self.circuit = VariationalQuantumCircuit(n_qubits, depth, backend)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(x, self.circuit, self.shift)

class QCNet(nn.Module):
    """CNN followed by a quantum hybrid head."""
    def __init__(self, n_qubits: int = 8, depth: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)

        self.quantum_head = QuantumHybridLayer(n_qubits, depth)
        self.final_fc = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_out = self.quantum_head(x)
        logit = self.final_fc(q_out)
        prob = torch.sigmoid(logit)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["VariationalQuantumCircuit", "QuantumHybridFunction",
           "QuantumHybridLayer", "QCNet"]
