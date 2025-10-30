import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import Aer, execute, transpile, assemble

class QuantumCircuit:
    """
    Wrapper around a parametrised circuit executed on a Qiskit simulator.
    The circuit consists of a Hadamard on every qubit followed by a
    rotation Ry(θ) where θ is a classical pixel value encoded as a
    parameter.  The expectation value of Pauli‑Z on each qubit is
    returned and averaged.
    """
    def __init__(self, n_qubits: int, backend, shots: int, threshold: float):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]

        # Build a universal circuit
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        for i in range(n_qubits):
            self._circuit.ry(self.theta[i], i)
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the parametrised circuit for the provided angles.

        Parameters
        ----------
        thetas : np.ndarray
            Shape (batch, n_qubits) containing the rotation angles.

        Returns
        -------
        np.ndarray
            Expectation value of Pauli‑Z averaged over all qubits.
        """
        # Prepare parameter bindings
        param_binds = []
        for sample in thetas:
            bind = {self.theta[i]: sample[i] for i in range(self.n_qubits)}
            param_binds.append(bind)

        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result().get_counts(self._circuit)

        # Compute expectation for each sample
        expectations = []
        if isinstance(result, list):
            # multiple executions (one per parameter bind)
            for counts in result:
                expectations.append(self._expectation(counts))
        else:
            expectations.append(self._expectation(result))

        return np.array(expectations)

    @staticmethod
    def _expectation(counts: dict) -> float:
        ones = 0
        total = 0
        for key, val in counts.items():
            ones += sum(int(bit) for bit in key) * val
            total += val
        if total == 0:
            return 0.0
        return ones / (total * len(next(iter(counts))))

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum circuit.
    Uses the parameter‑shift rule to compute gradients.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.cpu().numpy()
        exp = circuit.run(thetas)
        out = torch.tensor(exp, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for idx in range(inputs.shape[1]):
            inp_right = inputs.clone()
            inp_left = inputs.clone()
            inp_right[:, idx] += shift
            inp_left[:, idx] -= shift
            exp_right = ctx.circuit.run(inp_right.cpu().numpy())
            exp_left = ctx.circuit.run(inp_left.cpu().numpy())
            grads.append((exp_right - exp_left) / 2.0)
        grads = torch.tensor(grads, dtype=torch.float32, device=inputs.device).t()
        return grads * grad_output.unsqueeze(1), None, None

class ConvGen444(nn.Module):
    """
    Quantum‑augmented convolutional filter.
    The module extracts patches from the input image, flattens each
    patch into a vector of length kernel_size², and feeds that vector
    into a small variational quantum circuit.  The expectation value
    of the Pauli‑Z measurement is returned as the filter response.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 1,
                 threshold: float = 0.0,
                 shots: int = 100,
                 backend=None,
                 shift: float = np.pi / 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.shots = shots
        self.shift = shift
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2
        self.circuit = QuantumCircuit(self.n_qubits, self.backend, shots, threshold)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Input image of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Mean expectation value per batch element.
        """
        # Extract patches
        patches = torch.nn.functional.unfold(data, kernel_size=self.kernel_size,
                                             stride=self.stride)
        # patches shape (B, n_qubits, L)
        patches = patches.permute(0, 2, 1)  # (B, L, n_qubits)
        B, L, N = patches.shape
        flat = patches.reshape(-1, N)  # (B*L, N)
        # Convert to numpy for quantum circuit
        thetas = flat.cpu().numpy()
        # Encode pixel values into rotation angles
        thetas = np.where(thetas > self.threshold, np.pi, 0.0)
        exp = self.circuit.run(thetas)  # (B*L,)
        exp = torch.tensor(exp, dtype=torch.float32, device=data.device)
        exp = exp.reshape(B, L)
        return exp.mean(dim=1)

__all__ = ["ConvGen444"]
