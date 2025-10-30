"""Quantum fraud‑detection model using a parameterised two‑qubit circuit.

This module implements the same public API as the classical module but
replaces the deterministic head with a quantum expectation layer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumCircuitWrapper:
    """
    Parameterised two‑qubit circuit that maps a 2‑dimensional input
    vector to a single expectation value.
    """

    def __init__(self, backend: qiskit.providers.Backend = None, shots: int = 1024) -> None:
        self.backend = backend or AerSimulator()
        self.shots = shots

        # Build a generic circuit with two qubits
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        self.phi = qiskit.circuit.Parameter("phi")

        # Entangling layer
        self.circuit.h([0, 1])
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.phi, 1)
        self.circuit.cz(0, 1)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of 2‑dimensional parameters.

        Parameters
        ----------
        params : np.ndarray
            Shape (batch, 2) containing theta and phi for each instance.

        Returns
        -------
        np.ndarray
            Shape (batch,) expectation values of Z on qubit 0.
        """
        thetas = params[:, 0]
        phis = params[:, 1]
        expectations = []

        for theta, phi in zip(thetas, phis):
            bound = {self.theta: theta, self.phi: phi}
            transpiled = transpile(self.circuit, self.backend)
            bound_circ = transpiled.bind_parameters(bound)
            qobj = assemble(bound_circ, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(b, 2) for b in counts.keys()])
            # Expectation of Z on qubit 0: 1 for |0>, -1 for |1>
            exp_z = np.sum((1 - 2 * (states >> 0 & 1)) * probs)
            expectations.append(exp_z)

        return np.array(expectations)


class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between a PyTorch tensor and the quantum circuit.
    Uses a simple finite‑difference approximation for the gradient.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum: QuantumCircuitWrapper) -> torch.Tensor:
        ctx.quantum = quantum
        ctx.save_for_backward(inputs)
        # Run circuit on CPU to avoid device conflicts
        results = quantum.run(inputs.cpu().numpy())
        return torch.tensor(results, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        inputs, = ctx.saved_tensors
        quantum = ctx.quantum
        eps = 1e-3
        grad_inputs = []

        for i in range(inputs.shape[1]):
            unit = torch.zeros_like(inputs)
            unit[:, i] = eps
            plus = quantum.run((inputs + unit).cpu().numpy())
            minus = quantum.run((inputs - unit).cpu().numpy())
            grad = (plus - minus) / (2 * eps)
            grad_inputs.append(torch.tensor(grad, dtype=torch.float32, device=inputs.device))

        grad_inputs = torch.stack(grad_inputs, dim=1)
        return grad_output.unsqueeze(1) * grad_inputs, None


class FraudDetectionHybrid(nn.Module):
    """
    Quantum fraud‑detection model mirroring the classical interface.
    """

    def __init__(self, quantum: QuantumCircuitWrapper | None = None) -> None:
        super().__init__()
        self.quantum = quantum or QuantumCircuitWrapper()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Two‑column tensor with class probabilities.
        """
        logits = HybridFunction.apply(x, self.quantum)
        probs = self.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QuantumCircuitWrapper", "FraudDetectionHybrid", "HybridFunction"]
