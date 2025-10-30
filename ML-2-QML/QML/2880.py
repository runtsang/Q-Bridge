import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.providers.aer import AerSimulator

# ----------------------------------------------------------------------
#  Quantum utilities
# ----------------------------------------------------------------------
class QuantumCircuitWrapper:
    """
    Lightweight two‑qubit circuit that returns the expectation value of
    Pauli‑Z on the first qubit.  The circuit is parameterised by a single
    angle θ and is executed on the Aer simulator.
    """
    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, angles: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for a batch of angles.

        Parameters
        ----------
        angles : np.ndarray
            1‑D array of angles (θ) to bind to the circuit.

        Returns
        -------
        np.ndarray
            1‑D array of expectation values.
        """
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: float(a)} for a in angles],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Compute expectation value of Z: (+1) for |0>, (-1) for |1>
        exp_vals = []
        for count_dict in result if isinstance(result, list) else [result]:
            probs = np.array(list(count_dict.values())) / self.shots
            states = np.array(list(count_dict.keys())).astype(int)
            exp = np.sum(((-1) ** states) * probs)
            exp_vals.append(exp)
        return np.array(exp_vals)


# ----------------------------------------------------------------------
#  Autograd bridge between PyTorch and the quantum circuit
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that forwards activations through the quantum
    circuit and implements the parameter‑shift rule for gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float):
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            1‑D tensor of angles (θ) of shape (batch,).
        circuit : QuantumCircuitWrapper
            Quantum circuit to evaluate.
        shift : float
            Shift value for the parameter‑shift rule.

        Returns
        -------
        torch.Tensor
            1‑D tensor of expectation values of shape (batch,).
        """
        ctx.shift = shift
        ctx.circuit = circuit
        angles = inputs.detach().cpu().numpy()
        exp_vals = circuit.run(angles)
        return torch.tensor(exp_vals, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass implementing the parameter‑shift rule.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss w.r.t. the output of the forward pass.

        Returns
        -------
        Tuple[torch.Tensor, None, None]
            Gradient w.r.t. the inputs, and None for the other arguments.
        """
        inputs = ctx.saved_tensors[0] if ctx.saved_tensors else None
        shift = ctx.shift
        circuit = ctx.circuit
        angles = inputs.detach().cpu().numpy()
        grad_inputs = []
        for a in angles:
            exp_plus = circuit.run([a + shift])[0]
            exp_minus = circuit.run([a - shift])[0]
            grad = (exp_plus - exp_minus) / (2 * np.sin(shift))
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=grad_output.dtype, device=grad_output.device)
        return grad_inputs * grad_output, None, None


# ----------------------------------------------------------------------
#  Classical patch extractor (identical to the ML version)
# ----------------------------------------------------------------------
class ClassicalPatchFilter(nn.Module):
    """
    Classical 2×2 patch extractor implemented with a single Conv2d layer.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.conv(x)                     # (batch, 4, 14, 14)
        return patches.view(x.size(0), -1)         # (batch, 4*14*14)


# ----------------------------------------------------------------------
#  Hybrid quantum classifier
# ----------------------------------------------------------------------
class HybridQuantumClassifier(nn.Module):
    """
    Hybrid neural network that uses a classical patch extractor followed
    by a quantum expectation head.  The quantum head is a simple
    parameterised RY rotation on a single qubit; the expectation value
    is passed through a sigmoid to obtain a probability.
    """
    def __init__(self, shift: float = np.pi / 2, shots: int = 512):
        super().__init__()
        self.patch_filter = ClassicalPatchFilter()
        self.backend = AerSimulator()
        self.quantum_circuit = QuantumCircuitWrapper(self.backend, shots=shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Probability tensor of shape (batch, 2).
        """
        features = self.patch_filter(x)                     # (batch, 4*14*14)
        # Reduce to a single angle by averaging all patch features
        angles = features.mean(dim=1)                      # (batch,)
        # Quantum expectation via the differentiable wrapper
        exp_vals = HybridFunction.apply(angles, self.quantum_circuit, self.shift)
        probs = torch.sigmoid(exp_vals)                    # (batch,)
        return torch.cat([probs, 1 - probs], dim=-1)        # (batch, 2)

__all__ = ["HybridQuantumClassifier"]
