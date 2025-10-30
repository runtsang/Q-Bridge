import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator

class QuantumRegressionCircuit:
    """
    Two‑qubit parameterised circuit used as a regression head.
    The first qubit encodes the input via Ry, a CX entangles
    the qubits, and the second qubit is measured in the Z basis.
    """

    def __init__(self, shots: int = 2000) -> None:
        self.shots = shots
        self.backend = AerSimulator()
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circ = QC(2)
        theta = QC.Parameter("theta")
        phi = QC.Parameter("phi")

        self.circ.ry(theta, 0)
        self.circ.cx(0, 1)
        self.circ.rz(phi, 1)
        self.circ.cx(1, 0)
        self.circ.measure_all()

        self.params = {"theta": theta, "phi": phi}

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Return expectation value of Z on qubit 1 for each input."""
        bindings = [{"theta": float(x), "phi": 0.0} for x in inputs]
        compiled = transpile(self.circ, self.backend)
        qobj = assemble(compiled, shots=self.shots, parameter_binds=bindings)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()

        exp = []
        for key, val in counts.items():
            bit = int(key[1])  # qubit 1
            exp.append((1 - 2 * bit) * val)
        return np.array(exp) / self.shots

class QuantumHybridFunction(torch.autograd.Function):
    """
    Autograd wrapper that forwards a scalar through the quantum circuit.
    Uses the parameter‑shift rule for gradients.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumRegressionCircuit, shift: float = np.pi / 2) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp = circuit.run(inputs.detach().cpu().numpy())
        out = torch.tensor(exp, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad = []
        for x in inputs:
            right = ctx.circuit.run(np.array([x.item() + shift]))
            left = ctx.circuit.run(np.array([x.item() - shift]))
            grad.append((right - left) / (2 * shift))
        grad = torch.tensor(grad, device=grad_output.device, dtype=grad_output.dtype)
        return grad * grad_output, None, None

class EstimatorQNN(nn.Module):
    """
    Quantum regression model that uses a parameterised two‑qubit circuit
    as its output layer.  The model can be used standalone or as a head
    for a classical backbone.
    """

    def __init__(self, shots: int = 2000, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumRegressionCircuit(shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        return QuantumHybridFunction.apply(x, self.circuit, self.shift)

__all__ = ["EstimatorQNN"]
