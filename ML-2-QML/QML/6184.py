import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class VariationalCircuit:
    """
    Parametrised two‑qubit circuit with a repeatable RX‑RZ‑CNOT ansatz.
    """
    def __init__(self, n_qubits: int, layers: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.layers = layers
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.ParameterVector("theta", n_qubits * layers * 2)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        idx = 0
        for _ in range(self.layers):
            for q in range(self.n_qubits):          # RX rotations
                qc.rx(self.theta[idx], q)
                idx += 1
            for q in range(self.n_qubits):          # RZ rotations
                qc.rz(self.theta[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):      # CNOT ladder
                qc.cx(q, q + 1)
        qc.measure_all()
        return qc

    def run_batch(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each set of parameters in `thetas`.
        Thetas shape: (batch, num_params)
        Returns: expectations shape (batch,)
        """
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta[i]: val for i, val in enumerate(row)} for row in thetas]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts_list = result.get_counts()
        if isinstance(counts_list, dict):
            counts_list = [counts_list]
        expectations = []
        for counts in counts_list:
            total = sum(counts.values())
            exp = 0.0
            for state, cnt in counts.items():
                # Expectation of Z⊗...⊗Z via parity of the state string
                parity = (-1) ** (state.count("1"))
                exp += parity * cnt / total
            expectations.append(exp)
        return np.array(expectations)


class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that evaluates a batch of parameterised circuits
    and propagates gradients using the parameter shift rule.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: VariationalCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = circuit.run_batch(inputs.cpu().numpy())
        exp_tensor = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, exp_tensor)
        return exp_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        batch_size, n_params = inputs.shape
        grad_matrix = np.zeros((batch_size, n_params), dtype=np.float32)
        for idx in range(n_params):
            shift_vec = np.zeros_like(inputs.cpu().numpy())
            shift_vec[:, idx] = shift
            exp_plus = ctx.circuit.run_batch(inputs.cpu().numpy() + shift_vec)
            exp_minus = ctx.circuit.run_batch(inputs.cpu().numpy() - shift_vec)
            grad_matrix[:, idx] = (exp_plus - exp_minus) / 2.0
        grad_tensor = torch.tensor(grad_matrix, dtype=torch.float32, device=inputs.device)
        return grad_tensor * grad_output.unsqueeze(1), None, None


class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 200,
                 shift: float = np.pi / 2, layers: int = 2):
        super().__init__()
        if backend is None:
            backend = AerSimulator()
        self.quantum_circuit = VariationalCircuit(n_qubits, layers, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor):
        # inputs shape: (batch, n_qubits)
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)


class QCNet(nn.Module):
    """
    Convolutional network followed by a quantum expectation head.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(n_qubits=1, shots=200, shift=np.pi / 2, layers=2)

    def forward(self, inputs: torch.Tensor):
        x = F.relu(self.conv1(inputs))
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
        # Quantum hybrid head
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["VariationalCircuit", "HybridFunction", "Hybrid", "QCNet"]
