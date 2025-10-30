"""Quantum sampler network inspired by the original SamplerQNN, GraphQNN,
ClassicalQuantumBinaryClassification, and QuantumNAT.

The module defines a `SamplerQNNGen161` class that builds a parameterised
two‑qubit circuit with a user‑defined depth.  The circuit is executed on
Aer or any Qiskit backend and the resulting statevector is used as a
probability distribution over the computational basis.  The class
provides:

* Random unitary generation for each layer (`random_network`).
* Random training data generation (`random_training_data`).
* A hybrid expectation layer (`Hybrid`) that can be used as a
  differentiable head in larger models.
* Graph utilities (`fidelity_adjacency`) that map state fidelities to
  weighted adjacency graphs.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
import qutip as qt
import scipy as sc
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.providers.aer import Aer
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

# --------------------------------------------------------------------------- #
#  Core quantum utilities – adapted from GraphQNN.QML
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    identity = qt.qeye(dim)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Zero projector on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    projector = qt.fock(dim).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Swap qubits in a tensor product operator."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Generate a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (input, target) where the target is the
    unitary applied to the input."""
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    """Construct a random layered unitary network.

    Each layer consists of a single unitary that acts on the
    concatenation of the input qubits and a fresh ancilla for the
    output qubits.  The final target unitary is returned together
    with a training set for that unitary."""
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


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(
        layer_unitary * state * layer_unitary.dag(), range(num_inputs)
    )


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
) -> list[list[qt.Qobj]]:
    """Propagate a batch of states through the network."""
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
#  State fidelity and graph utilities – adapted from GraphQNN.QML
# --------------------------------------------------------------------------- #

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared absolute overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Hybrid quantum expectation layer – adapted from ClassicalQuantumBinaryClassification
# --------------------------------------------------------------------------- #

class HybridFunction(torch.autograd.Function):
    """Differentiable interface that evaluates a quantum expectation value
    for a batch of parameter sets."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        circuit: QuantumCircuit,
        backend,
        shots: int,
        shift: float,
    ) -> torch.Tensor:
        """Run the circuit for each set of parameters and return the expectation
        of the Z operator on the first qubit."""
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.backend = backend
        ctx.shots = shots

        # Prepare a list of parameter dictionaries
        param_binds = [
            {circuit.parameters[i]: val.item() for i, val in enumerate(row)}
            for row in inputs
        ]

        compiled = transpile(circuit, backend)
        qobj = assemble(compiled, shots=shots, parameter_binds=param_binds)
        job = backend.run(qobj)
        result = job.result()

        expectations = []
        for i in range(inputs.shape[0]):
            counts = result.get_counts(i)
            exp = 0.0
            total = 0
            for bitstring, ct in counts.items():
                # bitstring is string like '01'
                val = int(bitstring, 2)
                # Z expectation: +1 for |0>, -1 for |1>
                exp += ((-1) ** (val & 1)) * ct
                total += ct
            expectations.append(exp / total)
        ctx.save_for_backward(inputs)
        return torch.tensor(expectations, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Finite‑difference gradient approximation
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grad = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                perturbed = inputs.clone()
                perturbed[i, j] += shift
                out = HybridFunction.forward(
                    ctx, perturbed, ctx.circuit, ctx.backend, ctx.shots, shift
                )
                grad[i, j] = (out - ctx.saved_tensors[0][i, j]) / shift
        return grad, None, None, None, None


class Hybrid(nn.Module):
    """Wrapper that forwards a batch of parameters through a
    parameterised quantum circuit and returns the expectation value."""

    def __init__(self, n_qubits: int, backend, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift

        # Simple circuit: H on all qubits, parametric Ry, measure
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(ParameterVector("theta", n_qubits), range(n_qubits))
        self.circuit.measure_all()

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(params, self.circuit, self.backend, self.shots, self.shift)


# --------------------------------------------------------------------------- #
#  SamplerQNNGen161 – quantum version
# --------------------------------------------------------------------------- #

class SamplerQNNGen161:
    """Quantum sampler that mirrors the classical SamplerQNNGen161 but
    operates on quantum states.  The circuit is built from a stack of
    random unitaries (see `random_network`) and a final measurement
    that yields a probability distribution over the computational basis."""
    def __init__(
        self,
        qnn_arch: Sequence[int],
        backend=None,
        shots: int = 100,
        graph_threshold: float = 0.8,
        secondary: float | None = None,
    ) -> None:
        if backend is None:
            backend = Aer.get_backend("aer_simulator")
        self.backend = backend
        self.shots = shots
        self.graph_threshold = graph_threshold
        self.secondary = secondary

        # Build the random unitary network
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(
            qnn_arch, samples=100
        )

        # Build a Qiskit circuit that implements the same transformation
        self.circuit = QuantumCircuit(qnn_arch[-1])
        # Apply each layer's unitary as a sub‑circuit
        for layer, ops in enumerate(self.unitaries[1:], start=1):
            for op in ops:
                # Convert qutip Qobj to a Qiskit unitary
                unitary_matrix = op.full()
                sub = QuantumCircuit(qnn_arch[-1])
                sub.unitary(unitary_matrix, range(qnn_arch[-1]), label=f"layer{layer}")
                self.circuit.append(sub.to_instruction(), range(qnn_arch[-1]))
        self.circuit.measure_all()

        # Sampler primitive
        self.sampler = StatevectorSampler(backend=self.backend)

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of parameter sets and return
        the resulting probability vector for each sample."""
        # Build parameter bind list
        param_binds = [
            {self.circuit.parameters[i]: val for i, val in enumerate(row)}
            for row in inputs
        ]
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, parameter_binds=param_binds, shots=self.shots)
        job = self.backend.run(qobj)
        result = job.result()
        # Extract probabilities
        probs = []
        for i in range(inputs.shape[0]):
            counts = result.get_counts(i)
            prob = np.zeros(2 ** self.arch[-1])
            for bitstring, ct in counts.items():
                idx = int(bitstring, 2)
                prob[idx] = ct / self.shots
            probs.append(prob)
        return np.array(probs)

    def graph_of_states(self, samples: Iterable[tuple[qt.Qobj, qt.Qobj]]) -> nx.Graph:
        """Return a graph where nodes are the input states and edges reflect
        fidelity between the corresponding output states."""
        outputs = feedforward(self.arch, self.unitaries, samples)
        final_states = [layer[-1] for layer in outputs]
        return fidelity_adjacency(final_states, self.graph_threshold, secondary=self.secondary)

    @staticmethod
    def generate_random(qnn_arch: Sequence[int], samples: int):
        """Wrapper around GraphQNN.QML.random_network."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def generate_training_data(unitary: qt.Qobj, samples: int):
        """Wrapper around GraphQNN.QML.random_training_data."""
        return random_training_data(unitary, samples)

    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        """Return the squared overlap of two pure states."""
        return state_fidelity(a, b)

    __all__ = [
        "SamplerQNNGen161",
        "HybridFunction",
        "Hybrid",
        "random_network",
        "random_training_data",
        "feedforward",
        "state_fidelity",
        "fidelity_adjacency",
    ]
