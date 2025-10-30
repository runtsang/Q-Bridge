\
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile, execute
from qiskit import Aer


class QuantumCircuitWrapper:
    """Parameterized 2‑qubit circuit with entanglement and a Z‑expectation head."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        self.qc = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.ParameterVector("theta", 2 * n_qubits)

        for i in range(n_qubits):
            self.qc.h(i)
            self.qc.ry(self.theta[2 * i], i)
            self.qc.rz(self.theta[2 * i + 1], i)

        # Entangling layer
        for i in range(n_qubits - 1):
            self.qc.cx(i, i + 1)

        self.qc.measure_all()

    def _expectation(self, counts: dict) -> float:
        probs = {k: v / self.shots for k, v in counts.items()}
        exp = sum(((-1) ** int(bit[0])) * p for bit, p in probs.items())
        return exp

    def run(self, thetas: np.ndarray) -> np.ndarray:
        if thetas.ndim == 1:
            thetas = thetas[np.newaxis, :]
        batch = thetas.shape[0]

        compiled = transpile(self.qc, self.backend)
        circuits = []
        for params in thetas:
            bind = {p: params[i] for i, p in enumerate(self.theta)}
            bound = compiled.bind_parameters(bind)
            circuits.append(bound)

        job = execute(circuits, self.backend, shots=self.shots)
        results = job.result()
        expectations = [self._expectation(results.get_counts(idx)) for idx in range(batch)]
        return np.array(expectations, dtype=np.float32)


class HybridFunction(torch.autograd.Function):
    """Differentiable interface to the quantum circuit using parameter‑shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        shift_val = shift if shift!= 0 else 1e-3

        grads = []
        for idx in range(inputs.shape[-1]):
            inc = inputs.clone()
            dec = inputs.clone()
            inc[..., idx] += shift_val
            dec[..., idx] -= shift_val
            f_inc = circuit.run(inc.detach().cpu().numpy())
            f_dec = circuit.run(dec.detach().cpu().numpy())
            grad = (f_inc - f_dec) / (2 * shift_val)
            grads.append(grad)

        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device).T
        return grads * grad_output.unsqueeze(-1), None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)


class HybridQCNet(nn.Module):
    """Convolutional network followed by a quantum expectation head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout2d(0.3),
        )

        dummy = torch.zeros(1, 3, 32, 32)
        feature_dim = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Output 4 parameters for the 2‑qubit circuit
            nn.Linear(84, 4),
        )

        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits=2, backend=backend, shots=512, shift=np.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        probs = self.hybrid(x)
        return torch.cat([probs, 1 - probs], dim=-1)


__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid", "HybridQCNet"]
