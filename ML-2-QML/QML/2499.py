import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int):
    """
    Construct a layered ansatz with explicit encoding and variational parameters.
    Returns the circuit, encoding parameters, weight parameters, and observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = qiskit.QuantumCircuit(num_qubits)
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
    """
    Wrapper around a parametrised quantum circuit executed on Aer.
    """
    def __init__(self, num_qubits: int, depth: int, backend, shots: int):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = backend
        self.shots = shots

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[
                {enc: val for enc, val in zip(self.encoding, param)} for param in params
            ],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts_list = result.get_counts()
        def expectation(count_dict):
            probs = np.array(list(count_dict.values())) / self.shots
            states = np.array([int(k, 2) for k in count_dict.keys()])
            exps = []
            for i in range(len(self.observables)):
                bits = (states >> (len(self.observables) - 1 - i)) & 1
                exps.append(np.sum(probs * (1 - 2 * bits)))
            return np.array(exps)
        if isinstance(counts_list, list):
            return np.array([expectation(c) for c in counts_list])
        else:
            return np.array([expectation(counts_list)])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        exp_vals = ctx.circuit.run(inputs.numpy())
        result = torch.tensor(exp_vals, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        grads = []
        for i in range(inputs.shape[1]):
            shift_vec = np.zeros_like(inputs.numpy())
            shift_vec[:, i] = ctx.shift
            right = ctx.circuit.run((inputs.numpy() + shift_vec).tolist())
            left = ctx.circuit.run((inputs.numpy() - shift_vec).tolist())
            grads.append((right - left) / 2.0)
        grads = torch.tensor(grads).transpose(0, 1).float()
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """
    def __init__(self, num_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(num_qubits, depth=2, backend=backend, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)

class QCNet(nn.Module):
    """
    Convolutional network followed by a quantum expectation head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # map to 3 qubit parameters
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(num_qubits=3, backend=backend, shots=1024, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        exp_vals = self.hybrid(x)
        prob0 = torch.sigmoid(exp_vals[:, 0])
        prob1 = 1 - prob0
        return torch.stack((prob0, prob1), dim=-1)

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid", "QCNet"]
