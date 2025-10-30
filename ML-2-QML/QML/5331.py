"""
Quantum primitives for EstimatorQNNGen.

* QuantumCircuitWrapper – a parameterised multi‑qubit circuit
  that can be executed on any Qiskit backend.
* HybridQuantum – a torch.nn.Module that forwards a vector of
  qubit angles through QuantumCircuitWrapper and returns the
  expectation value of the Pauli‑Z tensor product.
* HybridQuantumHead – a torch.autograd.Function that implements
  the parameter‑shift rule for gradients.
* TransformerBlockQuantum – (optional) a transformer block that
  uses quantum modules for the attention and feed‑forward
  sub‑layers.  It is included for completeness but not used in
  the default EstimatorQNNGen.
"""

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator


class QuantumCircuitWrapper:
    """Parameterised circuit with `n_qubits` Ry gates followed by a
    global Hadamard layer.  The expectation value of the tensor
    product of Pauli‑Z on all qubits is returned.
    """

    def __init__(self, n_qubits: int, backend=None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(n_qubits)]

        # Apply a global Hadamard and parameterised Ry on each qubit
        self.circuit.h(range(n_qubits))
        for i, param in enumerate(self.theta):
            self.circuit.ry(param, i)
        self.circuit.barrier()

    def run(self, params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each row of *params*.

        Parameters
        ----------
        params : np.ndarray
            Shape (batch, n_qubits) or (n_qubits,).

        Returns
        -------
        np.ndarray
            Shape (batch, 1) – expectation value of Z⊗...⊗Z.
        """
        if params.ndim == 1:
            params = params.reshape(1, -1)

        expectations = []
        for row in params:
            bind = {param: val for param, val in zip(self.theta, row)}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[bind],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            expectation = 0.0
            for bitstring, cnt in counts.items():
                # (-1)^popcount(bitstring) is the eigenvalue of Z⊗...⊗Z
                sign = (-1) ** (bitstring.count("1"))
                expectation += sign * cnt
            expectation /= self.shots
            expectations.append(expectation)
        return np.array(expectations).reshape(-1, 1)


class HybridQuantumHead(torch.autograd.Function):
    """Autograd function that forwards a vector of qubit angles
    through QuantumCircuitWrapper and implements the parameter‑shift
    rule for gradients.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectations = ctx.circuit.run(inputs.detach().cpu().numpy())
        result = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.cpu().numpy()) * ctx.shift
        grads = []
        for idx, val in enumerate(inputs.cpu().numpy()):
            right = ctx.circuit.run([val + shift[idx]])
            left = ctx.circuit.run([val - shift[idx]])
            grads.append(right - left)
        grad_inputs = torch.tensor(grads, dtype=torch.float32, device=inputs.device) * grad_output
        return grad_inputs, None, None


class HybridQuantum(nn.Module):
    """Wrapper module that exposes the quantum head as a PyTorch layer."""

    def __init__(
        self,
        n_qubits: int,
        backend=None,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridQuantumHead.apply(inputs, self.circuit, self.shift)


# --------------------------------------------------------------------------- #
#  Optional transformer block that uses quantum modules
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(nn.Module):
    """A minimal transformer block that replaces the feed‑forward
    sub‑layer with a quantum module.  It is not required for the
    baseline EstimatorQNNGen but demonstrates how the quantum
    primitives can be inserted into a transformer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_ffn: int,
        n_qlayers: int = 1,
        q_device=None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.q_ffn = nn.Sequential(
            *[
                HybridQuantum(
                    n_qubits=n_qubits_ffn,
                )
                for _ in range(n_qlayers)
            ]
        )
        self.ffn_linear = nn.Sequential(
            nn.Linear(n_qubits_ffn, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Flatten to feed into quantum modules
        flat = x.view(x.size(0), -1)
        q_out = self.q_ffn(flat)
        q_out = self.ffn_linear(q_out)
        x = self.norm2(x + self.dropout(q_out.view_as(x)))
        return x


__all__ = [
    "QuantumCircuitWrapper",
    "HybridQuantumHead",
    "HybridQuantum",
    "TransformerBlockQuantum",
]
