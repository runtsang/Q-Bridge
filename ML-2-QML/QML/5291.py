import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter

# --------------------------------------------------------------------------- #
#  Quantum primitives
# --------------------------------------------------------------------------- #

class QuantumCircuitWrapper:
    """Parametrised two‑qubit circuit that returns the expectation of Z on qubit 0."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 400):
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots

        self.circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")

        # Simple entangling block
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a list of angle parameters."""
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            exp = 0.0
            for state, count in count_dict.items():
                bit = int(state[-1])  # qubit 0 is the least significant bit
                prob = count / self.shots
                exp += (1 - 2 * bit) * prob
            return exp

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# --------------------------------------------------------------------------- #
#  Sampler QNN wrapper
# --------------------------------------------------------------------------- #

class SamplerQNNWrapper:
    """A tiny neural network that mimics a quantum sampler’s output distribution."""
    def __init__(self):
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

# --------------------------------------------------------------------------- #
#  Hybrid autograd bridge
# --------------------------------------------------------------------------- #

class HybridFunction(torch.autograd.Function):
    """
    Differentiable interface that forwards a scalar tensor through a quantum circuit
    and returns the expectation value. Finite‑difference gradients are used.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float = np.pi / 2):
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.save_for_backward(inputs)

        exp = circuit.run(inputs.tolist())[0]
        return torch.tensor(exp, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit

        grads = []
        for val in inputs.tolist():
            right = circuit.run([val + shift])[0]
            left = circuit.run([val - shift])[0]
            grads.append(right - left)

        grads = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grads * grad_output, None, None

# --------------------------------------------------------------------------- #
#  Hybrid layer that plugs into a PyTorch model
# --------------------------------------------------------------------------- #

class HybridLayer(torch.nn.Module):
    """Wraps QuantumCircuitWrapper and HybridFunction into a single nn.Module."""
    def __init__(self, n_qubits: int = 2, shots: int = 200, shift: float = np.pi / 2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits=n_qubits, shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)

# --------------------------------------------------------------------------- #
#  Photonic fraud‑detection parameters (used by the classical module)
# --------------------------------------------------------------------------- #

from dataclasses import dataclass
from typing import Tuple

@dataclass
class FraudLayerParams:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

__all__ = [
    "QuantumCircuitWrapper",
    "SamplerQNNWrapper",
    "HybridFunction",
    "HybridLayer",
    "FraudLayerParams",
]
