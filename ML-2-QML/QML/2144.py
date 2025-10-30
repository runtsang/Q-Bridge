import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import qiskit
from qiskit import Aer, transpile, assemble
from qiskit.circuit import Parameter

class QuantumCircuitWrapper:
    """
    Parameterised variational circuit with entanglement.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend if backend else Aer.get_backend("aer_simulator")
        self.theta = Parameter("θ")
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            self._circuit.h(q)
        for q in range(n_qubits - 1):
            self._circuit.cx(q, q + 1)
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each set of parameters in thetas.
        """
        compiled = transpile(self._circuit, self.backend)
        results = []
        for theta in thetas:
            bound = compiled.bind_parameters({self.theta: theta})
            qobj = assemble(bound, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            exp = 0.0
            for state, prob in zip(counts.keys(), probs):
                z = 1 if state[0] == '0' else -1
                exp += z * prob
            results.append(exp)
        return np.array(results)

class HybridFunction(torch.autograd.Function):
    """
    Autograd function that forwards to the quantum circuit and
    implements the parameter‑shift rule for gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        inputs_np = inputs.detach().cpu().numpy()
        theta_expanded = np.stack([inputs_np] * circuit.n_qubits, axis=1)
        exp_vals = circuit.run(theta_expanded)
        result = torch.tensor(exp_vals, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for i in range(inputs.shape[0]):
            x = inputs[i].item()
            theta_plus = np.array([x + shift] * ctx.circuit.n_qubits)
            theta_minus = np.array([x - shift] * ctx.circuit.n_qubits)
            exp_plus = ctx.circuit.run(theta_plus.reshape(1, -1))[0]
            exp_minus = ctx.circuit.run(theta_minus.reshape(1, -1))[0]
            grad = (exp_plus - exp_minus) / 2
            grads.append(grad)
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """
    Wrapper that applies the HybridFunction to a scalar input.
    """
    def __init__(self, n_qubits: int = 2, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_squeezed = x.squeeze()
        return HybridFunction.apply(x_squeezed, self.circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """
    CNN backbone followed by a hybrid quantum head.
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
        self.hybrid = Hybrid(n_qubits=2, shift=np.pi / 2)

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
        x = self.fc3(x)
        x = self.hybrid(x).unsqueeze(-1)
        probs = torch.cat([x, 1 - x], dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            probs = self(x)
        return probs.argmax(dim=-1)
