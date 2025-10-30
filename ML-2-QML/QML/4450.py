"""Hybrid binary classifier – quantum implementation.

This module keeps the same CNN backbone but replaces the final classification head
with a variational quantum circuit (VQC).  An optional quantum‑kernel layer
can be inserted before the VQC, allowing experiments with kernel‑based quantum
feature maps.  The design is fully differentiable via the parameter‑shift rule.
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import assemble, transpile
import torchquantum as tq
from torchquantum.functional import func_name_dict


# --------------------------------------------------------------------------- #
#  Quantum circuit for the expectation head
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Two‑qubit parameterised circuit executed on Aer."""
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
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


# --------------------------------------------------------------------------- #
#  Differentiable bridge between PyTorch and the quantum circuit
# --------------------------------------------------------------------------- #
class HybridFunctionQuantum(torch.autograd.Function):
    """Forward the expectation value through the VQC."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift

        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None


# --------------------------------------------------------------------------- #
#  Quantum hybrid layer
# --------------------------------------------------------------------------- #
class HybridQuantum(nn.Module):
    """Layer that forwards activations through the VQC."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunctionQuantum.apply(squeezed, self.quantum_circuit, self.shift)


# --------------------------------------------------------------------------- #
#  Quantum kernel layer (optional)
# --------------------------------------------------------------------------- #
class QuantumKernelLayer(tq.QuantumModule):
    """TorchQuantum implementation of a quantum kernel."""
    def __init__(self, n_wires: int, prototypes: torch.Tensor):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.prototypes = prototypes  # shape (num_prototypes, n_wires)
        self.ansatz = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=x.device)
        # encode input
        self.ansatz(qdev, x)
        # compute overlaps with prototypes
        overlaps = []
        for proto in self.prototypes:
            qdev.reset_states(batch)
            self.ansatz(qdev, proto.unsqueeze(0).repeat(batch, 1))
            # overlap is absolute amplitude of |0...0>
            overlap = torch.abs(qdev.states.view(-1)[0])
            overlaps.append(overlap)
        return torch.stack(overlaps, dim=-1)  # shape (batch, num_prototypes)


# --------------------------------------------------------------------------- #
#  Quantum‑enhanced CNN backbone
# --------------------------------------------------------------------------- #
class QCNetQuantum(nn.Module):
    """CNN backbone identical to the classical version."""
    def __init__(self, n_qubits: int = 2) -> None:
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
        self.hybrid = HybridQuantum(n_qubits, backend, shots=100, shift=np.pi / 2)

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
        logits = self.fc3(x)                     # raw logits
        probs = self.hybrid(logits).T            # quantum expectation head
        return probs  # shape (batch, 1)


# --------------------------------------------------------------------------- #
#  Hybrid binary classifier with optional quantum kernel
# --------------------------------------------------------------------------- #
class HybridBinaryClassifierQuantum(nn.Module):
    """
    Quantum‑enhanced binary classifier.

    Parameters
    ----------
    use_kernel : bool
        If True, a quantum kernel layer is applied before the VQC head.
    n_qubits : int
        Number of qubits used in the VQC and kernel ansatz.
    """
    def __init__(self, use_kernel: bool = False, n_qubits: int = 2) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        self.cnn = QCNetQuantum(n_qubits)
        if use_kernel:
            self.kernel_layer: Optional[QuantumKernelLayer] = None
            self.kernel_classifier: Optional[nn.Linear] = None
        else:
            self.kernel_layer = None

    def set_kernel_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the prototype vectors for the quantum kernel."""
        if not self.use_kernel:
            raise RuntimeError("Kernel mapping is disabled.")
        self.kernel_layer = QuantumKernelLayer(prototypes.shape[1], prototypes)
        self.kernel_classifier = nn.Linear(prototypes.shape[0], 1).to(prototypes.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.cnn(x)  # shape (batch, 1)
        if self.use_kernel and self.kernel_layer is not None:
            kvecs = self.kernel_layer(logits)          # kernel features
            logits = self.kernel_classifier(kvecs)     # final linear layer
            probs = torch.sigmoid(logits)
            return torch.cat((probs, 1 - probs), dim=-1)
        return torch.cat((logits, 1 - logits), dim=-1)  # quantum head already outputs prob

__all__ = ["HybridBinaryClassifierQuantum"]
