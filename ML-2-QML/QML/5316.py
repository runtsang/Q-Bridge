import numpy as np
import qiskit
from qiskit import Aer, assemble, transpile
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumCircuit:
    """
    Parameterised two‑qubit circuit executed on Aer.
    The circuit implements a single Ry rotation on all qubits followed
    by measurement in the computational basis.  The expectation value
    of the Z Pauli operator is returned.
    """

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
        """Execute the parametrised circuit for the provided angles."""
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

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the QuantumCircuit.
    The forward pass evaluates the circuit; the backward pass
    numerically estimates the gradient via parameter‑shift.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.inputs = inputs
        expectation = circuit.run(inputs.tolist())
        return torch.tensor(expectation, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs = ctx.inputs
        shift = ctx.shift
        circuit = ctx.circuit
        grad = torch.zeros_like(inputs)
        for idx, val in enumerate(inputs):
            right = circuit.run([val.item() + shift])[0]
            left = circuit.run([val.item() - shift])[0]
            grad[idx] = (right - left) / 2.0
        return grad * grad_output, None, None

class Hybrid(nn.Module):
    """
    Wrapper that forwards activations through a QuantumCircuit.
    """

    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch,)
        return HybridFunction.apply(inputs, self.circuit, self.shift)

class UnifiedHybridLayer(nn.Module):
    """
    Hybrid layer that first extracts classical features and then
    evaluates them through a parameterised quantum circuit.  The class
    can be used as a drop‑in replacement for the classical FCL or
    QFCModel implementations from the reference seeds.
    """

    def __init__(self, in_channels: int = 1, mode: str = "cnn",
                 n_qubits: int = 2, shots: int = 100):
        super().__init__()
        if mode == "cnn":
            # CNN backbone inspired by QuantumNAT
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            self.norm = nn.BatchNorm1d(1)
        elif mode == "fcl":
            # Fully‑connected layer inspired by FCL
            self.linear = nn.Linear(1, 1)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        backend = Aer.get_backend("qasm_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "features"):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            out = self.norm(out)
        else:
            out = self.linear(x)
        # Pass through the quantum hybrid head
        # Ensure shape (batch,)
        out = out.squeeze(-1)
        return self.hybrid(out).unsqueeze(-1)
