"""Quantum utilities for hybrid models.

Provides a parameterised quantum circuit, an autograd Function for
back‑propagation, and a hybrid layer that can be plugged into a
classical network.  The module also bundles graph‑based utilities
(derived from the GraphQNN reference) so users can generate quantum
state embeddings from networkx graphs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import assemble, transpile
import qutip as qt
import networkx as nx
import scipy as sc
import itertools
from typing import Iterable, Sequence, Tuple

# Quantum circuit wrapper ----------------------------------------------------
class QuantumCircuitWrapper:
    """Parameterised, single‑shot circuit with Ry gates per qubit.

    The circuit prepares an equal superposition, applies a Ry rotation
    per qubit and measures in the computational basis.  The function
    returns the expectation value of Z for the first qubit, which
    serves as a scalar output for a quantum layer.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 100):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self.thetas = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n_qubits)]
        for i, theta in enumerate(self.thetas):
            self._circuit.ry(theta, i)
        self._circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots

    def _expectation(self, counts: dict) -> float:
        """Expectation value of Z on the first qubit."""
        probs = np.array(list(counts.values()), dtype=np.float64) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        z = (-1) ** ((states >> 0) & 1)  # first qubit
        return float(np.sum(z * probs))

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of theta vectors."""
        expectations = []
        for sample in thetas:
            compiled = transpile(self._circuit, self.backend)
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{theta: val for theta, val in zip(self.thetas, sample)}],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            expectations.append(self._expectation(counts))
        return np.array(expectations, dtype=np.float32)

# Autograd Function ---------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards a batch of parameters to a
    quantum circuit and returns the expectation value.  The backward
    pass uses a central‑difference estimator with a small step size.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float = 0.0):
        ctx.shift = shift
        ctx.circuit = circuit
        thetas = inputs.detach().cpu().numpy() + shift
        expectations = circuit.run(thetas)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        eps = 1e-4
        inputs_np = inputs.detach().cpu().numpy()
        grad = np.zeros_like(inputs_np)
        for i in range(inputs_np.shape[1]):
            plus = inputs_np.copy()
            minus = inputs_np.copy()
            plus[:, i] += eps
            minus[:, i] -= eps
            exp_plus = ctx.circuit.run(plus + shift)
            exp_minus = ctx.circuit.run(minus + shift)
            grad[:, i] = (exp_plus - exp_minus) / (2 * eps)
        grad = torch.tensor(grad, dtype=inputs.dtype, device=inputs.device)
        return grad * grad_output.unsqueeze(1), None, None

# Hybrid layer -------------------------------------------------------------
class HybridQuantumLayer(nn.Module):
    """A PyTorch layer that maps a vector of parameters to a quantum
    expectation value using the QuantumCircuitWrapper.
    """
    def __init__(self, n_qubits: int, backend=None, shots: int = 100, shift: float = 0.0):
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        exp = HybridFunction.apply(inputs, self.quantum_circuit, self.shift)
        # Map expectation [-1, 1] to probability [0, 1]
        return torch.sigmoid(exp)

# Graph‑based utilities ----------------------------------------------------
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int):
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states a and b."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity ≥ threshold receive weight 1.0.
    If ``secondary`` is provided, fidelities between ``secondary`` and
    ``threshold`` are added with ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "HybridQuantumLayer",
    "HybridFunction",
    "QuantumCircuitWrapper",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
