import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class ParametricTwoQubitCircuit:
    """
    Two‑qubit variational circuit used as the quantum head.
    Parameters:
        n_qubits (int): number of qubits (fixed to 2 for this head).
        backend (AerSimulator): qiskit Aer simulator.
        shots (int): number of measurement shots.
    """
    def __init__(self, n_qubits: int = 2, backend: AerSimulator = None, shots: int = 200) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")

        # Simple entangling block
        self.circuit.h(range(self.n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(self.n_qubits))
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a list of parameters.
        Returns the expectation value of Z on the first qubit.
        """
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts(self.circuit)

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        return np.array([expectation(result)])


class QuantumHeadFunction(torch.autograd.Function):
    """
    Autograd wrapper that maps a scalar input to a quantum expectation
    via the parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, circuit: ParametricTwoQubitCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Convert to numpy for circuit execution
        theta = input_tensor.cpu().detach().numpy()
        exp_val = circuit.run(theta)
        ctx.save_for_backward(input_tensor)
        return torch.tensor(exp_val, dtype=input_tensor.dtype, device=input_tensor.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_tensor, = ctx.saved_tensors
        shift = ctx.shift
        theta = input_tensor.cpu().detach().numpy()
        grad = []
        for t in theta:
            exp_plus = ctx.circuit.run([t + shift])
            exp_minus = ctx.circuit.run([t - shift])
            grad.append(exp_plus - exp_minus)
        grad = np.array(grad) / (2 * shift)
        return grad_output * torch.tensor(grad, dtype=grad_output.dtype, device=grad_output.device), None, None


class QuantumHead(nn.Module):
    """
    Fully‑connected layer that forwards through the quantum circuit.
    """
    def __init__(self, in_features: int, circuit: ParametricTwoQubitCircuit, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.shift = shift
        self.circuit = circuit
        self.param_mapper = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.param_mapper(x).squeeze(-1)
        return QuantumHeadFunction.apply(theta, self.circuit, self.shift)


class HybridClassifier(nn.Module):
    """
    CNN backbone with a quantum expectation head.
    Mirrors the classical version but replaces the final head with a
    quantum‑parameterised layer.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Quantum expectation head
        backend = AerSimulator()
        self.quantum_head = QuantumHead(self.fc3.out_features, ParametricTwoQubitCircuit(backend=backend), shift=np.pi / 2)

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
        x = self.quantum_head(x)
        return torch.cat((x, 1 - x), dim=-1)


__all__ = ["HybridClassifier"]
