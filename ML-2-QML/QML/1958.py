"""
Hybrid Quantum Binary Classifier – Quantum Backend.
Wraps a parameterised variational circuit executed on a chosen backend
(Qiskit Aer or Pennylane's default simulator). Provides a differentiable
hybrid layer that interfaces with PyTorch.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Qiskit; fall back to Pennylane if unavailable
try:
    import qiskit
    from qiskit import transpile, assemble
    from qiskit.providers.aer import AerSimulator
    _USE_QISKIT = True
except Exception:
    _USE_QISKIT = False
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        _USE_QISKIT = False
    except Exception:
        raise RuntimeError("Neither Qiskit nor Pennylane could be imported.")

# --------------------------------------------------------------------------- #
# 1.  Quantum circuit wrappers
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """
    Wrapper around a parameterised two‑qubit circuit executed on the chosen backend.
    The circuit applies H, RY(theta), and measures in the computational basis.
    """
    def __init__(self, n_qubits: int, backend, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        if _USE_QISKIT:
            self.backend = backend
            self.circuit = qiskit.QuantumCircuit(n_qubits)
            all_qubits = list(range(n_qubits))
            self.theta = qiskit.circuit.Parameter("theta")
            self.circuit.h(all_qubits)
            self.circuit.barrier()
            self.circuit.ry(self.theta, all_qubits)
            self.circuit.measure_all()
        else:
            self.backend = None
            self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
            @qml.qnode(self.dev, interface="torch")
            def circuit(theta):
                qml.Hadamard(wires=range(n_qubits))
                qml.RY(theta, wires=range(n_qubits))
                return qml.expval(qml.PauliZ(wires=0))
            self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the parametrised circuit for the provided angles.
        Returns the expectation value(s) for each theta.
        """
        if _USE_QISKIT:
            compiled = transpile(self.circuit, self.backend)
            exp = []
            for theta in thetas:
                qobj = assemble(
                    compiled,
                    shots=self.shots,
                    parameter_binds=[{self.circuit.parameters[0]: theta}],
                )
                job = self.backend.run(qobj)
                result = job.result()
                counts = result.get_counts()
                probs = np.array(list(counts.values())) / self.shots
                states = np.array([int(k, 2) for k in counts.keys()])
                exp.append(np.sum((1 - 2 * states) * probs))
            return np.array(exp)
        else:
            theta_tensor = torch.tensor(thetas, dtype=torch.float32, device="cpu")
            exp = self.circuit(theta_tensor)
            return exp.detach().cpu().numpy()

# --------------------------------------------------------------------------- #
# 2.  Hybrid autograd function
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    """
    Differentiable interface between PyTorch and the quantum circuit.
    Uses parameter‑shift rule in the backward pass.
    """
    @staticmethod
    def forward(ctx, logits: torch.Tensor, circuit: QuantumCircuit, shift: float = math.pi / 2) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = logits.detach().cpu().numpy()
        exp = circuit.run(thetas)
        probs = torch.tensor(exp, dtype=logits.dtype, device=logits.device)
        return probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        logits = grad_output.detach()
        shift = ctx.shift
        circuit = ctx.circuit
        thetas = logits.detach().cpu().numpy()
        exp_plus = circuit.run(thetas + shift)
        exp_minus = circuit.run(thetas - shift)
        grad = (exp_plus - exp_minus) / 2.0
        return grad * grad_output, None, None

# --------------------------------------------------------------------------- #
# 3.  Hybrid layer
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """
    def __init__(self, n_qubits: int, backend, shots: int = 1024, shift: float = math.pi / 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return HybridFunction.apply(flat, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
# 4.  Full model – CNN + quantum head
# --------------------------------------------------------------------------- #
class HybridQCNet(nn.Module):
    """
    Convolutional network followed by a quantum expectation head.
    An auxiliary classical head can be enabled for ensembling.
    """
    def __init__(self, backend, shots: int = 1024, aux: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2),

            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5),
        )
        self.flatten_dim = 15 * 13 * 13
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.hybrid = Hybrid(1, backend, shots=shots)
        self.aux = nn.Linear(84, 1) if aux else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        quantum_out = self.hybrid(x.unsqueeze(-1))
        if self.aux is not None:
            aux_out = torch.sigmoid(self.aux(x))
            out = (quantum_out + aux_out) / 2.0
        else:
            out = quantum_out
        return torch.cat([out, 1 - out], dim=-1)

__all__ = ["QuantumCircuit", "HybridFunction", "Hybrid", "HybridQCNet"]
