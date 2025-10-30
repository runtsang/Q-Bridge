import numpy as np
import qiskit
from qiskit import assemble, transpile
import torch
import torch.nn as nn

class QuantumCircuit:
    """
    Wrapper around a parametrised circuit executed on Aer.
    """
    def __init__(self, n_qubits: int, backend, shots: int):
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            counts_arr = np.array(list(counts.values()))
            states = np.array(list(counts.keys())).astype(float)
            probs = counts_arr / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable interface between PyTorch and the quantum circuit.
    Uses parameter‑shift rule for gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.squeeze().tolist()
        exp_vals = ctx.circuit.run(np.array(thetas))
        result = torch.tensor(exp_vals, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs) * ctx.shift
        grads = []
        for val, sh in zip(inputs.squeeze().tolist(), shift):
            exp_r = ctx.circuit.run(np.array([val + sh]))
            exp_l = ctx.circuit.run(np.array([val - sh]))
            grads.append(exp_r - exp_l)
        grad_tensor = torch.tensor(grads, dtype=torch.float32)
        return grad_tensor * grad_output, None, None

class HybridFullyConnectedLayer(nn.Module):
    """
    Quantum hybrid head that forwards activations through a parameterised circuit.
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : qiskit.providers.Backend
        Quantum backend (e.g., Aer simulator).
    shots : int
        Number of shots per execution.
    shift : float, optional
        Shift used in the parameter‑shift rule. Defaults to π/2.
    """
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.quantum = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs.squeeze(), self.quantum, self.shift)

class QCNet(nn.Module):
    """
    CNN-based binary classifier mirroring the structure of the quantum model,
    but using the quantum hybrid head.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = HybridFullyConnectedLayer(n_qubits, backend, shots, shift=np.pi/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.drop2(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x).T
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "HybridFullyConnectedLayer", "QCNet"]
