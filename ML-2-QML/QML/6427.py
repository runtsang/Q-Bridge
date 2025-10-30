"""Quantum‑enabled implementation of a hybrid convolutional classifier.

The network replaces the classical convolutional layers with a quantum
quanvolution filter implemented by a parameterised two‑qubit circuit.
A second quantum circuit serves as the expectation‑based binary head.
Both circuits are wrapped in differentiable PyTorch functions so that
the entire model can be trained end‑to‑end with gradient descent.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile, execute
from qiskit.circuit.random import random_circuit
import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum circuit for the expectation head
class QuantumCircuit:
    """Two‑qubit parameterised circuit returning an expectation value."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# Differentiable wrapper for the expectation circuit
class HybridFunctionQuantum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy().flatten()
        exp_vals = ctx.circuit.run(thetas)
        result = torch.tensor(exp_vals, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.detach().cpu().numpy()) * ctx.shift
        grads = []
        for val, s in zip(inputs.detach().cpu().numpy(), shift):
            right = ctx.circuit.run([val + s])[0]
            left = ctx.circuit.run([val - s])[0]
            grads.append(right - left)
        grad_inputs = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype) * grad_output
        return grad_inputs, None, None

# Quantum convolution (quanvolution) filter
class QuanvCircuit:
    """Parameterised circuit that acts as a 2‑D filter over a patch."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        # data shape: (kernel_size, kernel_size)
        data_flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)
        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class HybridQuantumClassifier(nn.Module):
    """Quantum‑enhanced CNN using a quanvolution filter and a quantum head."""
    def __init__(self,
                 kernel_size: int = 2,
                 conv_threshold: float = 127.0,
                 n_qubits_head: int = 2,
                 shift: float = np.pi / 2,
                 shots: int = 100) -> None:
        super().__init__()
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.quanv = QuanvCircuit(kernel_size, backend, shots, conv_threshold)
        self.quantum_head = QuantumCircuit(n_qubits_head, backend, shots)
        self.shift = shift
        self.shots = shots

        # Classical fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def _apply_quanv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum filter to each patch of the input."""
        batch, channels, h, w = x.shape
        # Convert to grayscale
        x = torch.mean(x, dim=1, keepdim=True)
        k = int(self.quanv.n_qubits ** 0.5)
        stride = k
        out_h = h // stride
        out_w = w // stride
        features = torch.zeros(batch, 1, out_h, out_w, device=x.device)
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, :, i*stride:(i+1)*stride, j*stride:(j+1)*stride]
                patch_np = patch.cpu().numpy()
                vals = []
                for b in range(batch):
                    val = self.quanv.run(patch_np[b, 0])
                    vals.append(val)
                features[:, 0, i, j] = torch.tensor(vals, device=x.device, dtype=x.dtype)
        return features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Apply quantum convolution filter
        qfeat = self._apply_quanv(inputs)
        # Flatten and feed into classical FC layers
        flat = torch.flatten(qfeat, 1)
        x = F.relu(self.fc1(flat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Quantum expectation head
        logits = HybridFunctionQuantum.apply(x.squeeze(), self.quantum_head, self.shift)
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = ["HybridQuantumClassifier", "QuantumCircuit", "QuanvCircuit", "HybridFunctionQuantum"]
