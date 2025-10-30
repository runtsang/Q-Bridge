"""Quantum hybrid binary classifier.

This module implements a convolutional backbone followed by a quantum head that
produces two expectation values corresponding to the two classes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector

class QuantumCircuitWrapper:
    """Parameterized quantum circuit with encoding and variational layers."""

    def __init__(self, num_qubits: int, depth: int, backend, shots: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend
        self.shots = shots

        # Parameters
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        self.circuit = QuantumCircuit(num_qubits)
        # Encoding
        for param, qubit in zip(self.encoding, range(num_qubits)):
            self.circuit.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.circuit.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.circuit.cz(qubit, qubit + 1)

        # Measurement
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Execute the circuit for an array of parameter vectors.

        Parameters
        ----------
        params : np.ndarray
            Shape (batch, num_qubits) containing the values to bind to the encoding parameters.

        Returns
        -------
        np.ndarray
            Shape (batch, num_qubits) of expectation values for each qubit.
        """
        expectations = []
        for p in params:
            bound = {self.encoding[i]: p[i] for i in range(self.num_qubits)}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots,
                            parameter_binds=[bound])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            exp = []
            for qubit in range(self.num_qubits):
                mask = 1 << (self.num_qubits - qubit - 1)
                pos = states & mask
                exp_val = (np.sum(probs[pos]) - np.sum(probs[~pos]))
                exp.append(exp_val)
            expectations.append(exp)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        params = inputs.detach().cpu().numpy()
        expectations = circuit.run(params)
        result = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = []
        for idx in range(inputs.shape[0]):
            p = inputs[idx].detach().cpu().numpy()
            pos = p + shift
            neg = p - shift
            exp_pos = ctx.circuit.run(pos.reshape(1, -1))[0]
            exp_neg = ctx.circuit.run(neg.reshape(1, -1))[0]
            grad = (exp_pos - exp_neg) / (2 * shift)
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=torch.float32, device=inputs.device)
        return grad_inputs * grad_output, None, None

class HybridBinaryClassifier(nn.Module):
    """Hybrid binary classifier with a convolutional backbone and a quantum head."""

    def __init__(self,
                 in_channels: int = 3,
                 num_qubits: int = 2,
                 depth: int = 2,
                 backend=None,
                 shots: int = 100,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_qubits),
            nn.ReLU(),
        )
        if backend is None:
            from qiskit import Aer
            backend = Aer.get_backend("aer_simulator")
        self.quantum_head = QuantumCircuitWrapper(num_qubits, depth, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        logits = HybridFunction.apply(features, self.quantum_head, self.shift)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridBinaryClassifier"]
