"""Quantum‑hybrid CNN with a variational circuit head and efficient evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit as QC, assemble, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List

class QuantumCircuitWrapper:
    """Parameterized two‑qubit circuit that returns expectation of Z on qubit 0."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = QC(n_qubits)
        self.theta = Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, 0)
        self._circuit.cx(0, 1)
        self._circuit.barrier()
        self._circuit.rz(self.theta, 1)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()), dtype=float)
            states = np.array([int(k, 2) for k in count_dict.keys()], dtype=float)
            probs = counts / self.shots
            # Expectation of Z on qubit 0: map 0->+1, 1->-1
            z_vals = np.where(states & 1, -1, 1)
            return np.sum(z_vals * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Bridge between PyTorch and the quantum circuit using the parameter‑shift rule."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        ctx.shift = shift
        ctx.circuit = circuit
        exp = circuit.run(inputs.tolist())
        out = torch.tensor(exp, dtype=torch.float32)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.numpy(), ctx.shift)
        grads = []
        for val in inputs.numpy():
            exp_plus = ctx.circuit.run([val + shift])
            exp_minus = ctx.circuit.run([val - shift])
            grads.append(exp_plus - exp_minus)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class HybridQuantumHead(nn.Module):
    """Hybrid head that evaluates a variational quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        return HybridFunction.apply(x, self.circuit, self.shift)

class HybridQuantumCNN(nn.Module):
    """CNN followed by a quantum expectation head with efficient evaluation."""
    def __init__(self, shots: int = 100, shift: float = np.pi / 2):
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
        self.hybrid_head = HybridQuantumHead(2, backend, shots, shift)

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
        probs = self.hybrid_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    def evaluate_with_shots(self, inputs: torch.Tensor, shots: int, seed: int | None = None) -> torch.Tensor:
        """Return predictions with explicit shot noise."""
        with torch.no_grad():
            probs = self.forward(inputs)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, 1 / np.sqrt(shots), size=probs.shape)
            return probs + noise

class FastBaseEstimator:
    """Convenience wrapper that evaluates a circuit for many parameter sets."""
    def __init__(self, circuit: QC):
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QC:
        mapping = dict(zip(self.params, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]):
        results = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            results.append([state.expectation_value(op) for op in observables])
        return results

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "HybridQuantumHead",
           "HybridQuantumCNN", "FastBaseEstimator"]
