import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, assemble, transpile

class EncoderBlock(nn.Module):
    """Same encoder as the classical variant."""
    def __init__(self, in_channels: int = 1, out_features: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_features, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_features)
        self.act = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class QuantumCircuit:
    """
    A shallow variational circuit with 4 qubits.
    Each qubit receives a rotation RY(theta_i) followed by a linear CNOT chain.
    The circuit is measured in the computational basis and the expectation
    value of Pauli‑Z on each qubit is returned.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend('aer_simulator')
        self.theta = [qiskit.circuit.Parameter(f'theta_{i}') for i in range(n_qubits)]
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()
    def _expectation_from_counts(self, counts: dict) -> np.ndarray:
        n = self.n_qubits
        exp = np.zeros(n)
        total = sum(counts.values())
        for state, count in counts.items():
            prob = count / total
            bits = [int(b) for b in state[::-1]]  # Qiskit uses little‑endian ordering
            for i, bit in enumerate(bits):
                exp[i] += (1 - 2 * bit) * prob
        return exp
    def run(self, thetas: np.ndarray) -> np.ndarray:
        thetas = np.asarray(thetas)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        batch = thetas.shape[0]
        compiled = transpile(self.circuit, self.backend, optimization_level=3)
        param_binds = [{p: val for p, val in zip(self.theta, row)} for row in thetas]
        qobj = assemble(compiled, parameter_binds=param_binds, shots=self.shots)
        job = self.backend.run(qobj)
        results = job.result()
        expectations = np.zeros((batch, self.n_qubits))
        for i, result in enumerate(results.get_counts()):
            expectations[i] = self._expectation_from_counts(result)
        return expectations

class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that evaluates the quantum circuit and
    implements the parameter‑shift rule in the backward pass.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = np.pi / 2):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.cpu().numpy()
        exp = circuit.run(thetas)  # shape (batch, n_qubits)
        out = torch.tensor(exp, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs)
        return out
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        thetas = inputs.cpu().numpy()
        n_params = thetas.shape[1]
        batch, _ = thetas.shape
        grad_thetas = np.zeros((batch, circuit.n_qubits, n_params))
        for i in range(n_params):
            plus = thetas.copy()
            minus = thetas.copy()
            plus[:, i] += shift
            minus[:, i] -= shift
            exp_plus = circuit.run(plus)
            exp_minus = circuit.run(minus)
            grad_thetas[:, :, i] = (exp_plus - exp_minus) / (2 * np.sin(shift))
        grad_input = torch.zeros_like(inputs)
        for i in range(n_params):
            grad_input[:, i] = torch.sum(grad_output * torch.tensor(grad_thetas[:, :, i], device=grad_output.device, dtype=grad_output.dtype), dim=1)
        return grad_input, None, None

class QuantumHybridNAT(nn.Module):
    """
    Quantum‑enhanced hybrid network.
    The convolutional backbone is identical to the classical variant.
    The final dense layer is replaced by a parameterised quantum circuit
    whose expectation values form the network output.
    """
    def __init__(self, in_channels: int = 1, n_qubits: int = 4, shift: float = np.pi / 2):
        super().__init__()
        self.encoder = EncoderBlock(in_channels, 16)
        self.features = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits),
        )
        self.quantum_circuit = QuantumCircuit(n_qubits=n_qubits)
        self.shift = shift
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.encoder(x)
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        params = self.fc(flat)  # shape (batch, n_qubits)
        out = HybridFunction.apply(params, self.quantum_circuit, self.shift)
        return out

__all__ = ["QuantumHybridNAT"]
