"""
Quantum regression module that replaces the final linear head with a quantum expectation layer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector

class QuantumCircuit:
    """
    Parameterised circuit mapping an array of angles to Pauli‑Z expectation values
    for each qubit.  Executed on the Qiskit Aer simulator.
    """
    def __init__(self, n_qubits: int, backend, shots: int):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        self.theta = ParameterVector("theta", n_qubits)
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each row of ``thetas`` (shape: batch × n_qubits)
        and return the expectation values of Pauli‑Z for each qubit.
        """
        if thetas.ndim == 1:
            thetas = thetas[None, :]
        batch = thetas.shape[0]
        expectations = np.zeros((batch, self.n_qubits), dtype=np.float32)

        compiled = transpile(self.circuit, self.backend)
        for i, theta_row in enumerate(thetas):
            param_bind = {self.theta[j]: theta_row[j] for j in range(self.n_qubits)}
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[param_bind],
            )
            job = self.backend.run(qobj)
            result = job.result().get_counts()

            probs = {int(k, 2): v / self.shots for k, v in result.items()}
            for q in range(self.n_qubits):
                exp = 0.0
                for bitstr, prob in probs.items():
                    bit = (bitstr >> q) & 1
                    exp += (1.0 if bit == 0 else -1.0) * prob
                expectations[i, q] = exp
        return expectations

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge that forwards angles to the quantum circuit and
    returns expectation values.  Gradients are computed with the parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, angles: torch.Tensor, circuit: QuantumCircuit, shift: float):
        ctx.circuit = circuit
        ctx.shift = shift
        angles_np = angles.detach().cpu().numpy()
        expectations_np = circuit.run(angles_np)
        expectations = torch.tensor(expectations_np, device=angles.device, dtype=torch.float32)
        ctx.save_for_backward(angles)
        return expectations

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        angles, = ctx.saved_tensors
        shift = ctx.shift
        grad = torch.zeros_like(angles)
        angles_np = angles.detach().cpu().numpy()
        for i in range(angles_np.shape[0]):  # batch
            for j in range(angles_np.shape[1]):  # qubit
                plus = angles_np.copy()
                minus = angles_np.copy()
                plus[i, j] += shift
                minus[i, j] -= shift
                plus_exp = ctx.circuit.run(plus)[i, j]
                minus_exp = ctx.circuit.run(minus)[i, j]
                grad[i, j] = (plus_exp - minus_exp) / (2 * shift)
        grad = torch.tensor(grad, device=angles.device, dtype=torch.float32)
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """
    PyTorch module that wraps the quantum circuit and exposes a differentiable forward pass.
    """
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(angles, self.quantum_circuit, self.shift)

class QuantumRegressor(nn.Module):
    """
    Hybrid regression model that maps raw features to angles, runs the quantum circuit,
    and uses a classical linear head to produce a scalar prediction.
    """
    def __init__(
        self,
        num_features: int,
        num_wires: int,
        backend=None,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_wires = num_wires
        self.feature_fc = nn.Linear(num_features, num_wires)
        self.hybrid = Hybrid(
            n_qubits=num_wires,
            backend=backend or qiskit.Aer.get_backend("aer_simulator"),
            shots=shots,
            shift=shift,
        )
        self.output_head = nn.Linear(num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = self.feature_fc(x)
        quantum_out = self.hybrid(angles)
        return self.output_head(quantum_out).squeeze(-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "QuantumRegressor"]
