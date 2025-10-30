"""Quantitative hybrid classifier – quantum implementation.

This module adds a quantum expectation head and a quantum‑based
self‑attention block on top of the classical backbone.  It uses
`qiskit` for circuit construction and execution, and
`torch.autograd.Function` to back‑propagate through the quantum
circuit.

Key components:
* `_QuantumCircuit`: a parameterised two‑qubit circuit that returns
  a single‑qubit expectation value.
* `HybridFunction`: autograd bridge between PyTorch and the quantum
  circuit.
* `HybridLayer`: a PyTorch module that forwards activations through
  the quantum circuit.
* `QuantumHybridClassifier`: the full model that can be toggled between
  classical and quantum heads via the ``use_quantum`` flag.

Typical usage::

    from QuantumHybridClassifier import QuantumHybridClassifier
    model = QuantumHybridClassifier(num_features=64, depth=3,
                                    attention_depth=2, use_quantum=True,
                                    n_qubits=4, quantum_backend="aer_simulator",
                                    shots=200)
    logits = model(torch.randn(8, 64))
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

__all__ = ["QuantumHybridClassifier"]


class _QuantumCircuit:
    """Two‑qubit parameterised circuit executed on an Aer backend.

    The circuit implements a simple variational ansatz:
    H → Ry(θ) → CZ → measure.
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
        """Execute the parametrised circuit for the provided angles.

        Parameters
        ----------
        thetas : np.ndarray
            Array of shape (batch,) containing angle values.

        Returns
        -------
        np.ndarray
            Array of shape (batch,) with expectation values.
        """
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
    """Differentiable interface between PyTorch and the quantum circuit.

    The forward pass evaluates the quantum circuit; the backward pass
    estimates gradients using parameter‑shift rules.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: _QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        values = inputs.cpu().numpy()
        expectations = circuit.run(values)
        return torch.tensor(expectations, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, shift, circuit = ctx.saved_tensors[0], ctx.shift, ctx.circuit
        grad_inputs = torch.zeros_like(inputs)
        shift_arr = np.full(inputs.shape, shift)
        # Parameter‑shift rule
        for idx in range(inputs.shape[0]):
            pos = inputs[idx] + shift_arr[idx]
            neg = inputs[idx] - shift_arr[idx]
            e_pos = circuit.run([pos.item()])[0]
            e_neg = circuit.run([neg.item()])[0]
            grad_inputs[idx] = (e_pos - e_neg) * grad_output[idx]
        return grad_inputs, None, None


class HybridLayer(nn.Module):
    """Layer that forwards activations through a quantum expectation."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = _QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


class _DenseEncoder(nn.Module):
    """Depth‑controlled dense encoder (identical to classical version)."""
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(num_features, num_features))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _QuantumSelfAttention(nn.Module):
    """Quantum self‑attention using a simple Qiskit circuit for each qubit.

    The implementation follows the reference `SelfAttention.py` quantum
    example.  The block encodes rotation parameters into X‑rotations,
    entangles adjacent qubits with controlled‑X, and measures.
    """
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")
        self.circuit = qiskit.QuantumCircuit(self.qr, self.cr)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circ = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def forward(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor) -> torch.Tensor:
        # Convert tensors to numpy for circuit construction
        rot_np = rotation_params.detach().cpu().numpy()
        ent_np = entangle_params.detach().cpu().numpy()
        circ = self._build_circuit(rot_np, ent_np)
        job = qiskit.execute(circ, self.backend, shots=self.shots)
        result = job.result().get_counts(circ)
        # Return measurement counts as a tensor
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        return torch.tensor(probs, device=rotation_params.device)


class QuantumHybridClassifier(nn.Module):
    """Hybrid classifier with optional quantum back‑end.

    Parameters
    ----------
    num_features : int
        Input feature dimensionality.
    depth : int
        Depth of the dense encoder.
    attention_depth : int
        Number of attention layers.
    use_quantum : bool
        If true, the model will use the quantum self‑attention and
        quantum hybrid head; otherwise it falls back to classical
        components.
    n_qubits : int
        Number of qubits for the quantum head and attention block.
    quantum_backend : str
        Name of the Qiskit Aer backend to use.
    shots : int
        Number of shots for quantum executions.
    shift : float
        Shift value for the parameter‑shift rule.
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 2,
                 attention_depth: int = 1,
                 use_quantum: bool = True,
                 n_qubits: int = 4,
                 quantum_backend: str = "aer_simulator",
                 shots: int = 200,
                 shift: float = math.pi / 2) -> None:
        super().__init__()
        self.use_quantum = use_quantum

        self.encoder = _DenseEncoder(num_features, depth)

        if self.use_quantum:
            backend = qiskit.Aer.get_backend(quantum_backend)
            self.attention_layers = nn.ModuleList(
                [_QuantumSelfAttention(n_qubits, backend, shots) for _ in range(attention_depth)]
            )
            self.hybrid = HybridLayer(n_qubits, backend, shots, shift)
        else:
            self.attention_layers = nn.ModuleList(
                [_DenseEncoder(num_features, 1) for _ in range(attention_depth)]
            )
            self.hybrid = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2).
        """
        x = self.encoder(x)
        if self.use_quantum:
            # Sample rotation and entangle parameters for each attention block
            batch_size = x.shape[0]
            for attn in self.attention_layers:
                rot = torch.randn(batch_size, 3 * attn.n_qubits, device=x.device)
                ent = torch.randn(batch_size, attn.n_qubits - 1, device=x.device)
                x = attn(rot, ent)
        else:
            for attn in self.attention_layers:
                x = attn(x)

        if self.use_quantum:
            logits = self.hybrid(x.squeeze(-1))
            logits = logits.unsqueeze(-1)
        else:
            logits = self.hybrid(x)

        return logits
