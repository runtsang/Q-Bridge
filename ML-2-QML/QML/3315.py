import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector

def build_classifier_circuit(num_qubits: int, depth: int):
    """Quantum ansatz with explicit encoding and variational layers."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class QuantumCircuitWrapper:
    """Runs a parametrised Qiskit circuit and returns expectation of Z."""
    def __init__(self, circuit: QuantumCircuit, backend=None, shots=1024):
        self.circuit = circuit
        self.backend = backend or AerSimulator()
        self.shots = shots

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        binds = [{p: val for p, val in zip(self.circuit.parameters, params)}]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=binds)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        def expectation(count_dict):
            probs = np.array(list(count_dict.values())) / self.shots
            states = np.array([int(k, 2) for k in count_dict.keys()])
            return np.sum(states * probs)
        return np.array([expectation(counts)])

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        inputs_np = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(inputs_np)
        result = torch.tensor(exp_vals, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.detach().cpu().numpy()) * ctx.shift
        grads = []
        for val, s in zip(inputs.detach().cpu().numpy(), shift):
            right = ctx.circuit.run([val + s])
            left = ctx.circuit.run([val - s])
            grads.append(right - left)
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Quantum hybrid layer that forwards activations through a variational circuit."""
    def __init__(self, num_qubits: int, backend=None, shots=1024, shift=np.pi / 2):
        super().__init__()
        circuit, _, _, _ = build_classifier_circuit(num_qubits, depth=2)
        self.wrapper = QuantumCircuitWrapper(circuit, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.wrapper, self.shift)

class HybridQuantumClassifier(nn.Module):
    """Convolutional backbone followed by a quantum expectation head."""
    def __init__(self, num_features=3, depth=2, num_qubits=2, backend=None, shots=1024, shift=np.pi / 2):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        # Output num_qubits parameters for the quantum circuit
        self.fc3 = nn.Linear(84, num_qubits)
        self.hybrid = Hybrid(num_qubits, backend, shots, shift)

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
        q_out = self.hybrid(x).squeeze(-1)
        probs = torch.cat((q_out, 1 - q_out), dim=-1)
        return probs

__all__ = ["HybridQuantumClassifier"]
