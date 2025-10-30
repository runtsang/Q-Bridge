import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Tuple, List
from qiskit import Aer, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational parameters.
    Returns the circuit, encoding parameters, weight parameters, and observables.
    """
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

class QuantumHybridFunction(torch.autograd.Function):
    """
    Differentiable interface that runs the parametrised quantum circuit for each
    sample in the batch and returns the averaged Z‑observable expectation as a
    probability. Gradients w.r.t. the input angles are estimated with the
    central finite‑difference rule.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, weight_params: List[nn.Parameter], shift: float, backend, shots: int) -> torch.Tensor:
        ctx.shift = shift
        ctx.backend = backend
        ctx.shots = shots
        ctx.circuit = circuit
        ctx.weight_params = weight_params

        batch_size = inputs.shape[0]
        probs = []
        for i in range(batch_size):
            input_vals = inputs[i].detach().cpu().numpy()
            binding = {}
            # Bind encoding parameters
            for param, val in zip(circuit.parameters[:circuit.num_qubits], input_vals):
                binding[param] = val
            # Bind variational weights
            for param, val in zip(circuit.parameters[circuit.num_qubits:], weight_params):
                binding[param] = val.item()
            compiled = transpile(circuit, backend)
            qobj = assemble(compiled, shots=shots, parameter_binds=[binding])
            result = backend.run(qobj).result()
            counts = result.get_counts()
            exp_z = 0.0
            for state, cnt in counts.items():
                prob = cnt / shots
                for q in range(circuit.num_qubits):
                    bit = int(state[::-1][q])  # little‑endian
                    exp_z += (1 - 2 * bit) * prob / circuit.num_qubits
            prob = (1 + exp_z) / 2.0
            probs.append(prob)
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return probs_tensor.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        backend = ctx.backend
        shots = ctx.shots
        circuit = ctx.circuit
        weight_params = ctx.weight_params
        batch_size = inputs.shape[0]
        grad_inputs = torch.zeros_like(inputs)
        for i in range(batch_size):
            input_vals = inputs[i].detach().cpu().numpy()
            grad = []
            for idx in range(len(input_vals)):
                # Right shift
                right = input_vals.copy()
                right[idx] += shift
                binding_r = {}
                for param, val in zip(circuit.parameters[:circuit.num_qubits], right):
                    binding_r[param] = val
                for param, val in zip(circuit.parameters[circuit.num_qubits:], weight_params):
                    binding_r[param] = val.item()
                compiled = transpile(circuit, backend)
                qobj_r = assemble(compiled, shots=shots, parameter_binds=[binding_r])
                result_r = backend.run(qobj_r).result()
                counts_r = result_r.get_counts()
                exp_r = 0.0
                for state, cnt in counts_r.items():
                    prob = cnt / shots
                    for q in range(circuit.num_qubits):
                        bit = int(state[::-1][q])
                        exp_r += (1 - 2 * bit) * prob / circuit.num_qubits
                # Left shift
                left = input_vals.copy()
                left[idx] -= shift
                binding_l = {}
                for param, val in zip(circuit.parameters[:circuit.num_qubits], left):
                    binding_l[param] = val
                for param, val in zip(circuit.parameters[circuit.num_qubits:], weight_params):
                    binding_l[param] = val.item()
                compiled = transpile(circuit, backend)
                qobj_l = assemble(compiled, shots=shots, parameter_binds=[binding_l])
                result_l = backend.run(qobj_l).result()
                counts_l = result_l.get_counts()
                exp_l = 0.0
                for state, cnt in counts_l.items():
                    prob = cnt / shots
                    for q in range(circuit.num_qubits):
                        bit = int(state[::-1][q])
                        exp_l += (1 - 2 * bit) * prob / circuit.num_qubits
                grad.append(exp_r - exp_l)
            grad = torch.tensor(grad, dtype=torch.float32, device=inputs.device)
            grad_inputs[i] = grad
        return grad_inputs * grad_output.squeeze(-1), None, None, None, None, None

class QuantumHybridHead(nn.Module):
    """
    Wrapper that exposes the quantum circuit as a differentiable layer.
    """
    def __init__(self, num_qubits: int, depth: int, shift: float = np.pi / 2, shots: int = 100):
        super().__init__()
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.shift = shift
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        # Convert weight parameters to nn.Parameter for optimisation
        self.weight_params = nn.ParameterList([nn.Parameter(torch.rand(1)) for _ in self.weights])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return QuantumHybridFunction.apply(inputs, self.circuit, list(self.weight_params), self.shift, self.backend, self.shots)

class HybridBinaryClassifier(nn.Module):
    """
    Convolutional network followed by a quantum expectation head. The design
    mirrors the classical counterpart but replaces the dense head with a
    variational circuit whose parameters are learned jointly.
    """
    def __init__(self, depth: int = 3, num_qubits: int = 4, shift: float = np.pi / 2, shots: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_qubits)
        self.hybrid = QuantumHybridHead(num_qubits, depth, shift=shift, shots=shots)

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
        probs = self.hybrid(x).squeeze(-1)
        return torch.stack((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
