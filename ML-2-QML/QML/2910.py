"""
Quantum‑parameterized circuit for refining latent codes in a hybrid auto‑encoder.
The circuit uses a RealAmplitudes ansatz followed by a swap‑test with a fixed reference
state. The expectation value of the auxiliary qubit is returned as a differentiable
scalar and can be used as a refinement of the classical latent representation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler import transpile
from qiskit.compiler import assemble
from qiskit.providers.aer import AerSimulator

# ------------------------------------------------------------------
# Quantum circuit wrapper
# ------------------------------------------------------------------
class QuantumAutoencoderCircuit(nn.Module):
    """
    Variational circuit that takes a latent vector as input parameters,
    applies a RealAmplitudes ansatz, and performs a swap test with a
    fixed reference state. The output is the expectation value of the
    auxiliary qubit, which can be interpreted as a quantum distance
    metric.
    """
    def __init__(self, num_latent: int, num_trash: int = 2, reps: int = 3):
        super().__init__()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.backend = AerSimulator()
        self.shots = 1024

        # Build the parametric circuit
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Parameterised ansatz on latent + trash qubits
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps)
        qc.append(ansatz, list(range(self.num_latent + self.num_trash)))

        # Swap test with a fixed reference state (|0...0⟩)
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def _bind_params(self, theta: np.ndarray) -> QuantumCircuit:
        """Bind a numpy array of parameters to the circuit."""
        param_dict = {f"theta_{i}": th for i, th in enumerate(theta)}
        return self._circuit.bind_parameters(param_dict)

    def evaluate(self, params: np.ndarray) -> float:
        """Run the circuit for a single set of parameters and return expectation."""
        # Ensure params are flattened
        theta = params.reshape(-1)
        circ = self._bind_params(theta)
        circ = transpile(circ, self.backend)
        qobj = assemble(circ, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        # Compute expectation of Z on the auxiliary qubit
        exp = 0.0
        for state, cnt in counts.items():
            exp += (1 if state == '1' else -1) * cnt
        return exp / self.shots

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Accepts a batch of latent vectors and returns a refined latent
        representation via the quantum expectation value. The operation
        is wrapped in a custom autograd function to allow gradient flow.
        """
        return QuantumRefineFunction.apply(latent, self)


class QuantumRefineFunction(Function):
    """
    Autograd wrapper that forwards a batch of latent vectors to the
    QuantumAutoencoderCircuit and returns the expectation value.
    Gradients are computed using the parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, latent: torch.Tensor, circuit: QuantumAutoencoderCircuit) -> torch.Tensor:
        ctx.circuit = circuit
        # Convert to numpy for evaluation
        latent_np = latent.detach().cpu().numpy()
        # Evaluate each sample in the batch
        results = np.array([circuit.evaluate(vec) for vec in latent_np])
        # Return as a column vector
        return torch.tensor(results, dtype=torch.float32, device=latent.device).unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Parameter‑shift rule: compute gradient for each latent dimension
        latents = ctx.saved_tensors[0] if ctx.saved_tensors else None
        if latents is None:
            # If not saved, we cannot compute gradient; return zeros
            return torch.zeros_like(grad_output), None

        shift = np.pi / 2
        grad_latent = []
        for i in range(latents.size(1)):
            plus = latents.clone()
            minus = latents.clone()
            plus[:, i] += shift
            minus[:, i] -= shift
            exp_plus = ctx.circuit.evaluate_batch(plus.detach().cpu().numpy())
            exp_minus = ctx.circuit.evaluate_batch(minus.detach().cpu().numpy())
            grad_i = (exp_plus - exp_minus) / 2.0
            grad_latent.append(grad_i.reshape(-1, 1))
        grad_latent = torch.tensor(np.concatenate(grad_latent, axis=1), dtype=torch.float32)
        return grad_output * grad_latent, None


# ------------------------------------------------------------------
# Convenience wrapper for batch evaluation
# ------------------------------------------------------------------
def evaluate_batch(circuit: QuantumAutoencoderCircuit, batch: np.ndarray) -> np.ndarray:
    """Vectorised evaluation for a batch of latent vectors."""
    return np.array([circuit.evaluate(vec) for vec in batch])

QuantumAutoencoderCircuit.evaluate_batch = evaluate_batch

__all__ = ["QuantumAutoencoderCircuit", "QuantumRefineFunction"]
