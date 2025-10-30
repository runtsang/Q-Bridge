import torch
import torch.nn as nn
import numpy as np
import qiskit
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class VariationalCircuit:
    'Simple two‑qubit variational circuit with a single rotation parameter.'
    def __init__(self, n_qubits: int = 2, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.circuit = QC(n_qubits)
        self.theta = qiskit.circuit.Parameter('theta')
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, 0)
        self.circuit.barrier()
        self.circuit.measure_all()
        self.backend = AerSimulator()
        self.compiled = transpile(self.circuit, self.backend)

    def expectation(self, theta: float) -> float:
        bound = self.compiled.bind_parameters({self.theta: theta})
        qobj = assemble(bound, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        exp = 0.0
        for bitstring, count in counts.items():
            bit = int(bitstring[0])
            prob = count / self.shots
            exp += (1 if bit == 0 else -1) * prob
        return exp

class QuantumExpectation(torch.autograd.Function):
    'Differentiable interface that returns the expectation value of the circuit.'
    @staticmethod
    def forward(ctx, angles: torch.Tensor, circuit: VariationalCircuit, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.save_for_backward(angles)
        exp_vals = []
        for a in angles.detach().cpu().numpy():
            exp_vals.append(circuit.expectation(float(a)))
        return torch.tensor(exp_vals, device=angles.device, dtype=angles.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        angles, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for a in angles.detach().cpu().numpy():
            exp_plus = circuit.expectation(a + shift)
            exp_minus = circuit.expectation(a - shift)
            grads.append(exp_plus - exp_minus)
        grad_tensor = torch.tensor(grads, device=angles.device, dtype=angles.dtype)
        return grad_output * grad_tensor, None, None

class QuantumHybridClassifier(nn.Module):
    'Hybrid quantum classifier that maps features to a rotation angle and returns a probability.'
    def __init__(self, in_features: int, n_qubits: int = 2, shots: int = 1024,
                 shift: float = np.pi / 2, trainable: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.classical = nn.Linear(in_features, 1, bias=False)
        self.classical.weight.requires_grad = trainable
        self.circuit = VariationalCircuit(n_qubits=n_qubits, shots=shots)
        self.shift = shift
        self.trainable = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1]!= self.in_features:
            raise ValueError(f'Expected last dim {self.in_features}, got {x.shape[-1]}')
        angle = self.classical(x).squeeze(-1)
        exp = QuantumExpectation.apply(angle, self.circuit, self.shift)
        probs = torch.sigmoid(exp)
        return torch.cat([probs, 1 - probs], dim=-1)

    def reset_params(self) -> None:
        'Reinitialize classical layer weights and reset quantum circuit parameters.'
        nn.init.constant_(self.classical.weight, 0.0)

__all__ = ['QuantumHybridClassifier', 'VariationalCircuit', 'QuantumExpectation']
