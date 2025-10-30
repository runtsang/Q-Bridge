import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """
    Two‑qubit parameterised circuit that applies a Ry rotation with a
    trainable angle θ to each qubit and measures the Z expectation value
    on the first qubit.  Supports batched execution and a parameter‑shift
    rule for gradient estimation.
    """
    def __init__(self, backend: AerSimulator = None, shots: int = 200):
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.theta = Parameter("θ")

        self.circuit = qiskit.QuantumCircuit(2)
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for an array of angles.  `thetas` is a 1‑D
        array of shape (batch,).  The function returns a 1‑D array of
        expectation values for the first qubit.
        """
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{self.theta: float(t)} for t in thetas]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()

        expectations = []
        for counts in result.get_counts():
            probs = {k: v / self.shots for k, v in counts.items()}
            exp = 0.0
            for state, p in probs.items():
                # Qiskit uses little‑endian bit order: last char = qubit 0
                bit = int(state[-1])
                exp += (1 - 2 * bit) * p
            expectations.append(exp)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """
    Torch autograd Function that forwards the input through the quantum
    circuit and implements the parameter‑shift rule for gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.save_for_backward(inputs)
        thetas = inputs.detach().cpu().numpy()
        exp_values = ctx.circuit.run(thetas)
        return torch.tensor(exp_values, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        pos_thetas = (inputs + shift).detach().cpu().numpy()
        neg_thetas = (inputs - shift).detach().cpu().numpy()
        f_plus = ctx.circuit.run(pos_thetas)
        f_minus = ctx.circuit.run(neg_thetas)
        grad_inputs = (f_plus - f_minus) / (2 * shift)
        return torch.tensor(grad_inputs, dtype=grad_output.dtype, device=grad_output.device) * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through the quantum circuit.
    """
    def __init__(self, shift: float = np.pi / 2):
        super().__init__()
        self.shift = shift
        self.circuit = QuantumCircuit()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(-1)
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class QCNet(nn.Module):
    """
    Convolutional network followed by a quantum expectation head.
    The hybrid head now supports batched execution and a trainable
    rotation angle per input feature.  The final output is a two‑class
    probability distribution produced by a softmax.
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

        self.hybrid = Hybrid()
        self.prob_head = nn.Linear(1, 2)

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

        q_expect = self.hybrid(x.squeeze(-1))
        q_expect = q_expect.unsqueeze(-1)

        logits = self.prob_head(q_expect)
        probs = F.softmax(logits, dim=-1)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience inference method.
        """
        self.eval()
        with torch.no_grad():
            return self(x)

__all__ = ["QCNet"]
