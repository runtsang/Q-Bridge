"""HybridQuantumCNN – quantum expectation head for binary classification.

The quantum head implements a 3‑qubit variational circuit with a single
parameter per qubit.  The expectation value of Z on qubit 0 is returned
as the probability of class 1.  Gradients are computed via the
parameter‑shift rule, making the head fully differentiable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumCircuit:
    """Parameterised 3‑qubit ansatz executed on the Aer simulator."""

    def __init__(self, shots: int = 1024):
        self.shots = shots
        self.backend = AerSimulator()
        self._circuit = qiskit.QuantumCircuit(3)
        self.theta0 = qiskit.circuit.Parameter("theta0")
        self.theta1 = qiskit.circuit.Parameter("theta1")
        self.theta2 = qiskit.circuit.Parameter("theta2")

        # Build a simple layered ansatz
        self._circuit.h([0, 1, 2])
        self._circuit.ry(self.theta0, 0)
        self._circuit.ry(self.theta1, 1)
        self._circuit.ry(self.theta2, 2)
        # Entangling layer
        self._circuit.cx(0, 1)
        self._circuit.cx(1, 2)
        # Second layer
        self._circuit.ry(self.theta0, 0)
        self._circuit.ry(self.theta1, 1)
        self._circuit.ry(self.theta2, 2)
        # Measurement
        self._circuit.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        """
        angles: shape (batch, 3)
        Returns expectation of Z on qubit 0 for each run.
        """
        compiled = transpile(self._circuit, self.backend)
        expectations = []
        for angle_vec in angles:
            param_bindings = {
                self.theta0: angle_vec[0],
                self.theta1: angle_vec[1],
                self.theta2: angle_vec[2]
            }
            bound_circ = compiled.bind_parameters(param_bindings)
            job = self.backend.run(assemble(bound_circ, shots=self.shots))
            result = job.result()
            counts = result.get_counts()
            # Compute expectation of Z on qubit 0
            exp = 0.0
            for bitstring, cnt in counts.items():
                bit0 = int(bitstring[-1])  # qubit 0 is LSB
                exp += ((-1) ** bit0) * cnt
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations, dtype=np.float32)

class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper around the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = np.pi / 2) -> torch.Tensor:
        """
        Forward pass that evaluates the quantum expectation.

        Parameters
        ----------
        inputs
            Tensor of shape (batch, 3) containing the angles.
        circuit
            Instance of QuantumCircuit.
        shift
            Shift value for the parameter‑shift rule.
        Returns
        -------
        Tensor of shape (batch, 1) with the expectation values.
        """
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)
        angles = inputs.detach().cpu().numpy()
        expectations = circuit.run(angles)
        return torch.tensor(expectations, device=inputs.device, dtype=inputs.dtype).unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using the parameter‑shift rule.
        """
        shift = ctx.shift
        circuit = ctx.circuit
        inputs, = ctx.saved_tensors
        batch, num_params = inputs.shape
        grad_inputs = torch.zeros_like(inputs)
        angles_np = inputs.detach().cpu().numpy()
        for b in range(batch):
            for i in range(num_params):
                angle_plus = angles_np[b].copy()
                angle_minus = angles_np[b].copy()
                angle_plus[i] += shift
                angle_minus[i] -= shift
                exp_plus = circuit.run(angle_plus.reshape(1, -1))[0]
                exp_minus = circuit.run(angle_minus.reshape(1, -1))[0]
                grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
                grad_inputs[b, i] = grad
        grad_inputs = grad_inputs.to(grad_output.device)
        grad_inputs = grad_inputs * grad_output
        return grad_inputs, None, None

class HybridLayer(nn.Module):
    """Layer that forwards activations through the quantum circuit."""

    def __init__(self, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class HybridQuantumCNN(nn.Module):
    """Quantum expectation head that returns a probability for class 1."""

    def __init__(self, shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_layer = HybridLayer(shots=shots, shift=shift)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        angles
            Tensor of shape (batch, 3) containing angles for the 3 qubits.
        Returns
        -------
        Tensor of shape (batch, 1) with the probability of class 1.
        """
        expectation = self.quantum_layer(angles)
        prob = torch.sigmoid(expectation)
        return prob

__all__ = ["QuantumCircuit", "HybridFunction", "HybridLayer", "HybridQuantumCNN"]
