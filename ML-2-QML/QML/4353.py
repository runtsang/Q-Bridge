import torch
import torch.nn as nn
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.quantum_info import SparsePauliOp
from qiskit import Aer

def build_classifier_circuit(num_qubits: int, depth: int):
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = [f"x_{i}" for i in range(num_qubits)]
    weights = [f"theta_{i}" for i in range(num_qubits * depth)]
    circuit = QuantumCircuit(num_qubits)
    for i, q in enumerate(range(num_qubits)):
        circuit.rx(encoding[i], q)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, encoding, weights, observables

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, backend, shots: int, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.backend = backend
        ctx.shots = shots
        expectation = []
        for params in inputs:
            compiled = transpile(circuit, backend)
            qobj = assemble(compiled, shots=shots,
                            parameter_binds=[{p: v} for p, v in zip(circuit.parameters, params.tolist())])
            job = backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp_val = 0.0
            for state, count in counts.items():
                prob = count / shots
                exp_val += int(state, 2) * prob
            expectation.append(exp_val)
        exp_tensor = torch.tensor(expectation, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, exp_tensor)
        return exp_tensor
    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for idx, val in enumerate(inputs):
            right = ctx.circuit.copy()
            left = ctx.circuit.copy()
            # Right shift
            params = list(right.parameters)
            params[idx] = val + shift
            right.assign_parameters(dict(zip(right.parameters, params)))
            # Left shift
            params = list(left.parameters)
            params[idx] = val - shift
            left.assign_parameters(dict(zip(left.parameters, params)))
            right_exp = HybridFunction.forward(None, right, ctx.backend, ctx.shots, shift)
            left_exp = HybridFunction.forward(None, left, ctx.backend, ctx.shots, shift)
            grad_inputs.append((right_exp - left_exp).item())
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None, None, None

class HybridQuantumClassifier(nn.Module):
    """
    Hybrid neural network that forwards activations through a variational quantum circuit.
    """
    def __init__(self, num_qubits: int, depth: int = 2, shots: int = 100, shift: float = np.pi / 2):
        super().__init__()
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.backend = Aer.get_backend("aer_simulator")
        self.shots = shots
        self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        exp_values = HybridFunction.apply(inputs, self.circuit, self.backend, self.shots, self.shift)
        probs = torch.sigmoid(exp_values)
        return probs

__all__ = ["HybridQuantumClassifier"]
