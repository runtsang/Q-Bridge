"""UnifiedSamplerGraphRegressor – quantum side of the hybrid sampler‑regression model.

This module implements the quantum counterpart of
``UnifiedSamplerGraphRegressor`` from the classical module.  It builds a
parameterised variational circuit whose connectivity follows the adjacency
graph produced by the classical feed‑forward network.  The circuit outputs
both a two‑class probability distribution (via a softmax over measurement
counts) and a regression value (via the expectation of Pauli‑Z).  Gradient
estimation uses the parameter‑shift rule, allowing end‑to‑end training with
PyTorch autograd.

Key components:
* ``QuantumSamplerCircuit`` – constructs a circuit with a configurable
  number of qubits and a graph‑based entangling pattern.
* ``HybridQuantumFunction`` – a custom autograd function that runs the
  circuit and returns the expectation value.
* ``UnifiedSamplerGraphRegressorQuantum`` – combines the sampler and
  regression heads into a single ``torch.nn.Module``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from qiskit.result import Result
from qiskit.quantum_info import Statevector
import networkx as nx
import itertools
import math

# Helper: generate a random unitary for each edge
def random_unitary(num_qubits: int) -> np.ndarray:
    """Return a Haar‑random unitary matrix."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q

class QuantumSamplerCircuit:
    """Parameterised circuit whose entanglement follows a graph."""
    def __init__(self, n_qubits: int, graph: nx.Graph, backend: Backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.graph = graph
        self.backend = backend
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.params = ParameterVector("θ", length=2 * n_qubits + self.graph.number_of_edges())
        self._build()

    def _build(self):
        # Input rotations
        for q in range(self.n_qubits):
            self.circuit.ry(self.params[2 * q], q)
        # Entangling layer following the graph
        edge_params = self.params[2 * self.n_qubits :]
        idx = 0
        for (u, v) in self.graph.edges():
            self.circuit.cx(u, v)
            self.circuit.ry(edge_params[idx], u)
            self.circuit.cx(u, v)
            idx += 1
        # Final local rotations
        for q in range(self.n_qubits):
            self.circuit.ry(self.params[2 * q + self.graph.number_of_edges()], q)
        self.circuit.measure_all()

    def run(self, param_values: np.ndarray) -> np.ndarray:
        """Execute the circuit for the given parameter vector."""
        bound_circuit = self.circuit.bind_parameters(dict(zip(self.params, param_values)))
        transpiled = transpile(bound_circuit, self.backend)
        qobj = assemble(transpiled, shots=self.shots)
        result: Result = self.backend.run(qobj).result()
        counts = result.get_counts()
        # Convert counts to expectation of Z on each qubit
        expectations = []
        for q in range(self.n_qubits):
            exp = 0.0
            for bitstring, cnt in counts.items():
                bit = int(bitstring[::-1][q])  # Qiskit ordering
                exp += (1 if bit == 0 else -1) * cnt
            exp /= self.shots
            expectations.append(exp)
        return np.array(expectations)

class HybridQuantumFunction(torch.autograd.Function):
    """Differentiable interface to the quantum sampler."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumSamplerCircuit, shift: float) -> torch.Tensor:
        # inputs: (batch, param_len)
        batch, param_len = inputs.shape
        expectations = []
        for i in range(batch):
            param_vals = inputs[i].detach().cpu().numpy()
            exp = circuit.run(param_vals)
            expectations.append(exp)
        exp_tensor = torch.tensor(expectations, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs)
        ctx.circuit = circuit
        ctx.shift = shift
        return exp_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        batch, param_len = inputs.shape
        grad_inputs = torch.zeros_like(inputs)
        shift = ctx.shift
        # Parameter‑shift rule
        for p in range(param_len):
            shift_vec = torch.zeros(param_len, dtype=inputs.dtype, device=inputs.device)
            shift_vec[p] = shift
            plus = inputs + shift_vec
            minus = inputs - shift_vec
            exp_plus = []
            exp_minus = []
            for i in range(batch):
                exp_plus.append(ctx.circuit.run(plus[i].detach().cpu().numpy()))
                exp_minus.append(ctx.circuit.run(minus[i].detach().cpu().numpy()))
            exp_plus = torch.tensor(exp_plus, dtype=torch.float32, device=inputs.device)
            exp_minus = torch.tensor(exp_minus, dtype=torch.float32, device=inputs.device)
            grad = (exp_plus - exp_minus) / (2 * shift)
            grad_inputs[:, p] = (grad_output * grad).sum(dim=0)
        return grad_inputs, None, None

class UnifiedSamplerGraphRegressorQuantum(nn.Module):
    """Quantum hybrid model mirroring the classical UnifiedSamplerGraphRegressor."""
    def __init__(
        self,
        n_qubits: int,
        graph: nx.Graph,
        backend: Backend = None,
        shots: int = 1024,
        shift: float = math.pi / 2,
    ) -> None:
        super().__init__()
        if backend is None:
            backend = Aer.get_backend("aer_simulator")
        self.circuit = QuantumSamplerCircuit(n_qubits, graph, backend, shots)
        self.shift = shift
        # Regression head: linear mapping from expectation vector to scalar
        self.regressor = nn.Linear(n_qubits, 1)
        # Sampler head: softmax over two outputs derived from two selected qubits
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, param_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        param_batch: (batch, param_len)
        Returns:
            probs: (batch, 2) – softmax over two classifier logits
            reg: (batch,) – sigmoid‑scaled regression value
        """
        expectations = HybridQuantumFunction.apply(param_batch, self.circuit, self.shift)
        logits = self.classifier(expectations)
        probs = F.softmax(logits, dim=-1)
        raw_reg = self.regressor(expectations).squeeze(-1)
        reg = torch.sigmoid(raw_reg)
        return probs, reg

__all__ = ["UnifiedSamplerGraphRegressorQuantum", "HybridQuantumFunction", "QuantumSamplerCircuit"]
