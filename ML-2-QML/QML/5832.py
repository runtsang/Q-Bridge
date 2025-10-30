"""UnifiedHybridEstimator: quantum‑enabled hybrid model for regression and classification.

It combines the EstimatorQNN feed‑forward regressor with the QCNet quantum‑head.
The quantum head uses a two‑qubit parameterised circuit executed on AerSimulator.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import AerSimulator

__all__ = ["QuantumCircuitWrapper", "HybridFunctionQuantum", "HybridQuantum", "UnifiedHybridEstimator"]

class QuantumCircuitWrapper:
    """Parameterised two‑qubit circuit that accepts an input and a weight."""
    def __init__(self, backend: AerSimulator, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(2)
        self.input_param = qiskit.circuit.Parameter("inp")
        self.weight_param = qiskit.circuit.Parameter("wgt")

        # Simple variational ansatz
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.input_param, 0)
        self.circuit.ry(self.weight_param, 1)
        self.circuit.measure_all()

        # Observable: Y on first qubit
        self.observable = qiskit.quantum_info.SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of (input, weight) pairs."""
        compiled = transpile(self.circuit, self.backend)
        expectations = []
        for theta in thetas:
            bind = {self.input_param: theta[0], self.weight_param: theta[1]}
            qobj = assemble(compiled, parameter_binds=[bind], shots=self.shots)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            # Expectation of Y: +1 for |1>, -1 for |0>
            exp = 0.0
            for state, cnt in counts.items():
                y_val = 1 if state[0] == '1' else -1
                exp += y_val * cnt
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

class HybridFunctionQuantum(torch.autograd.Function):
    """Forward pass runs the quantum circuit and returns the expectation.
    Backward uses the parameter‑shift rule for each input dimension."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float = np.pi / 2):
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.inputs = inputs.detach().cpu().numpy()
        expectations = ctx.circuit.run(ctx.inputs)
        return torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        inputs = ctx.inputs
        batch_size, dim = inputs.shape
        grad_inputs = np.zeros_like(inputs)
        for i in range(batch_size):
            for d in range(dim):
                theta_plus = inputs[i].copy()
                theta_plus[d] += shift
                theta_minus = inputs[i].copy()
                theta_minus[d] -= shift
                exp_plus = circuit.run(np.array([theta_plus]))[0]
                exp_minus = circuit.run(np.array([theta_minus]))[0]
                grad_inputs[i, d] = 0.5 * (exp_plus - exp_minus)
        grad_tensor = torch.tensor(grad_inputs, dtype=grad_output.dtype, device=grad_output.device)
        # Multiply by upstream gradient
        return grad_tensor * grad_output.squeeze(-1).unsqueeze(-1), None, None

class HybridQuantum(nn.Module):
    """Quantum hybrid head that forwards the linear logits through a quantum circuit."""
    def __init__(self, in_features: int, shift: float = np.pi / 2):
        super().__init__()
        self.linear = nn.Linear(in_features, 2)  # produce (input, weight)
        self.shift = shift
        self.backend = AerSimulator()
        self.circuit = QuantumCircuitWrapper(self.backend, shots=512)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        params = self.linear(inputs)
        return HybridFunctionQuantum.apply(params, self.circuit, self.shift)

class UnifiedHybridEstimator(nn.Module):
    """CNN backbone followed by a quantum hybrid head."""
    def __init__(self, output_dim: int = 1, shift: float = np.pi / 2):
        super().__init__()
        self.output_dim = output_dim
        self.shift = shift

        # CNN backbone identical to classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.hybrid = HybridQuantum(self.fc3.out_features, shift=self.shift)

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
        logits = self.hybrid(x)

        if self.output_dim == 1:
            return logits
        else:
            probs = torch.sigmoid(logits)
            return torch.cat([probs, 1 - probs], dim=-1)
