"""Quantum self‑attention module with a hybrid expectation head.

This module defines a single class `UnifiedSelfAttentionHybrid` that
provides a self‑attention block implemented as a parameterised quantum
circuit and a differentiable hybrid head that uses a second quantum
circuit to compute an expectation value.  The class follows the same
`run` interface as the classical counterpart and can be dropped into
training pipelines that use PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble, transpile
from qiskit.providers import Backend

class QuantumSelfAttentionCircuit:
    """Parametrised circuit that implements a quantum self‑attention block."""
    def __init__(self, n_qubits: int, backend: Backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

    def build(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        for i in range(self.n_qubits):
            self.circuit.rx(rotation_params[3*i], i)
            self.circuit.ry(rotation_params[3*i+1], i)
            self.circuit.rz(rotation_params[3*i+2], i)
        for i in range(self.n_qubits-1):
            self.circuit.crx(entangle_params[i], i, i+1)
        self.circuit.measure(self.qr, self.cr)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        self.circuit = QuantumCircuit(self.qr, self.cr)
        self.build(rotation_params, entangle_params)
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            prob = cnt / self.shots
            for i in range(self.n_qubits):
                if state[self.n_qubits-1-i] == '1':
                    probs[i] += prob
        return probs

class QuantumHybridCircuit:
    """Simple expectation circuit used as a hybrid head."""
    def __init__(self, n_qubits: int, backend: Backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.circuit = QuantumCircuit(self.qr)
        self.theta = qiskit.circuit.Parameter("theta")
        for i in range(n_qubits):
            self.circuit.ry(self.theta, i)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: p} for p in params])
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts(self.circuit)
        def expectation(count_dict):
            probs = np.array(list(count_dict.values())) / self.shots
            states = np.array([int(k, 2) for k in count_dict.keys()])
            return np.sum(states * probs)
        return np.array([expectation(counts)])

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumHybridCircuit, shift: float = 0.0):
        ctx.shift = shift
        ctx.circuit = circuit
        with torch.no_grad():
            input_np = inputs.detach().cpu().numpy()
            expectation = ctx.circuit.run(input_np)
        return torch.tensor(expectation, dtype=inputs.dtype, device=inputs.device)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        grad = []
        input_np = grad_output.detach().cpu().numpy()
        for val in input_np:
            e_plus = circuit.run(np.array([val + shift]))
            e_minus = circuit.run(np.array([val - shift]))
            grad.append(e_plus - e_minus)
        grad = torch.tensor(grad, dtype=grad_output.dtype, device=grad_output.device)
        return grad * grad_output, None, None

class UnifiedSelfAttentionHybrid:
    """Quantum self‑attention block with a differentiable hybrid head."""
    def __init__(self, embed_dim: int, backend: Backend, shots: int = 1024, shift: float = np.pi/2):
        self.embed_dim = embed_dim
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.attention_circuit = QuantumSelfAttentionCircuit(embed_dim, backend, shots)
        self.hybrid_circuit = QuantumHybridCircuit(embed_dim, backend, shots)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Compute attention weights via a quantum circuit, weight the inputs,
        and apply a hybrid quantum expectation head.

        Parameters
        ----------
        rotation_params : np.ndarray
            Flat array of length 3*embed_dim containing rotation angles.
        entangle_params : np.ndarray
            Flat array of length embed_dim-1 containing entanglement angles.
        inputs : np.ndarray
            Input tensor of shape (batch, embed_dim).

        Returns
        -------
        np.ndarray
            Output of shape (batch, 2) containing a probability pair
            (p, 1‑p) for binary classification.
        """
        attn_weights = self.attention_circuit.run(rotation_params, entangle_params)
        weighted = attn_weights @ inputs
        expectations = []
        for w in weighted:
            expectation = self.hybrid_circuit.run(np.array([w]))
            expectations.append(expectation)
        expectations = np.array(expectations).reshape(-1, 1)
        probs = 1 / (1 + np.exp(-expectations))
        return np.concatenate((probs, 1 - probs), axis=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(flat, self.hybrid_circuit, self.shift)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embed_dim={self.embed_dim})"

__all__ = ["UnifiedSelfAttentionHybrid", "HybridFunction"]
