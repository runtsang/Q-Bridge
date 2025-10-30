"""Hybrid classical self‑attention layer with a quantum attention weight generator.

The design keeps the original SelfAttention signature but expands the
attention mechanism:
  * `embed_dim` and `seq_len` are used to create linear projections
    and a quantum circuit with `seq_len` qubits.
  * `rotation_params` and `entangle_params` are trainable PyTorch
    tensors that drive a Qiskit circuit.
  * `QuantumAttentionFunction` implements a custom autograd
    function that evaluates the circuit and applies the
    parameter‑shift rule for gradients.
  * The module accepts batched input `(B, T, D)` and returns the
    attended output `(B, T, D)`.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.opflow import PauliExpectation, StateFn, CircuitStateFn, PauliOp, Pauli

class QuantumAttentionFunction(Function):
    """Autograd wrapper for a quantum circuit that outputs a
    probability distribution over `n_qubits` tokens via the
    expectation values of Pauli‑Z on each qubit.
    """

    @staticmethod
    def forward(ctx, rotation_params, entangle_params, n_qubits, backend):
        ctx.save_for_backward(rotation_params, entangle_params)
        ctx.n_qubits = n_qubits
        ctx.backend = backend

        # Build circuit
        qr = QuantumRegister(n_qubits, "q")
        circuit = QuantumCircuit(qr)
        for i in range(n_qubits):
            circuit.rx(rotation_params[3 * i].item(), i)
            circuit.ry(rotation_params[3 * i + 1].item(), i)
            circuit.rz(rotation_params[3 * i + 2].item(), i)
        for i in range(n_qubits - 1):
            circuit.crx(entangle_params[i].item(), i, i + 1)

        # Compute expectation of Z on each qubit
        probs = []
        for i in range(n_qubits):
            pauli_str = ["I"] * n_qubits
            pauli_str[i] = "Z"
            op = PauliOp(Pauli("".join(pauli_str)))
            state = StateFn(op, wrap=True)
            circuit_state = CircuitStateFn(circuit)
            exp_val = PauliExpectation().convert(state @ circuit_state)
            result = exp_val.eval(backend=backend)
            prob = 0.5 * (result + 1.0)
            probs.append(prob)

        probs = torch.tensor(probs, dtype=torch.float32, device=rotation_params.device)
        probs = torch.softmax(probs, dim=-1)
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        rotation_params, entangle_params = ctx.saved_tensors
        n_qubits = ctx.n_qubits
        backend = ctx.backend

        def eval_params(rot, ent):
            qr = QuantumRegister(n_qubits, "q")
            circuit = QuantumCircuit(qr)
            for i in range(n_qubits):
                circuit.rx(rot[3 * i].item(), i)
                circuit.ry(rot[3 * i + 1].item(), i)
                circuit.rz(rot[3 * i + 2].item(), i)
            for i in range(n_qubits - 1):
                circuit.crx(ent[i].item(), i, i + 1)

            probs = []
            for i in range(n_qubits):
                pauli_str = ["I"] * n_qubits
                pauli_str[i] = "Z"
                op = PauliOp(Pauli("".join(pauli_str)))
                state = StateFn(op, wrap=True)
                circuit_state = CircuitStateFn(circuit)
                exp_val = PauliExpectation().convert(state @ circuit_state)
                result = exp_val.eval(backend=backend)
                prob = 0.5 * (result + 1.0)
                probs.append(prob)
            probs = torch.tensor(probs, dtype=torch.float32, device=rot.device)
            probs = torch.softmax(probs, dim=-1)
            return probs

        shift = np.pi / 2
        grad_rot = torch.zeros_like(rotation_params)
        for idx in range(rotation_params.shape[0]):
            rot_plus = rotation_params.clone()
            rot_minus = rotation_params.clone()
            rot_plus[idx] += shift
            rot_minus[idx] -= shift
            probs_plus = eval_params(rot_plus, entangle_params)
            probs_minus = eval_params(rot_minus, entangle_params)
            grad_rot[idx] = 0.5 * torch.sum((probs_plus - probs_minus) * grad_output)

        grad_ent = torch.zeros_like(entangle_params)
        for idx in range(entangle_params.shape[0]):
            ent_plus = entangle_params.clone()
            ent_minus = entangle_params.clone()
            ent_plus[idx] += shift
            ent_minus[idx] -= shift
            probs_plus = eval_params(rotation_params, ent_plus)
            probs_minus = eval_params(rotation_params, ent_minus)
            grad_ent[idx] = 0.5 * torch.sum((probs_plus - probs_minus) * grad_output)

        return grad_rot, grad_ent, None, None

class SelfAttention(nn.Module):
    """Hybrid self‑attention module.

    Parameters
    ----------
    embed_dim : int
        Dimension of input embeddings.
    seq_len : int
        Number of tokens in the sequence (must equal the number of qubits).
    backend : qiskit.providers.Backend, optional
        Quantum backend. Defaults to the Aer qasm simulator.
    """

    def __init__(self, embed_dim: int, seq_len: int, backend=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Quantum parameters
        self.rotation_params = nn.Parameter(
            torch.randn(3 * seq_len, requires_grad=True)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(seq_len - 1, requires_grad=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, T, D) where T == seq_len.

        Returns
        -------
        torch.Tensor
            Output of shape (B, T, D).
        """
        B, T, D = x.shape
        assert T == self.seq_len, "Sequence length must match quantum circuit size"

        # Linear projections
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)

        # Compute quantum attention distribution once per batch
        attn_weights = QuantumAttentionFunction.apply(
            self.rotation_params, self.entangle_params, self.seq_len, self.backend
        )  # (T,)

        # Expand to (B, T, T) for broadcasting
        attn_weights = attn_weights.unsqueeze(0).expand(B, -1, -1)  # (B, T, T)

        # Compute attention output
        out = torch.bmm(attn_weights, v)  # (B, T, D)
        return out

__all__ = ["SelfAttention"]
