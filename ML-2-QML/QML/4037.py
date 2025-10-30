"""
Quantum implementation of the hybrid classifier.
Provides a parameterised circuit, utilities to generate synthetic data,
and a wrapper that runs the circuit on a simulator and feeds the
expectation values into a classical PyTorch classifier.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable

import torch
import torch.nn as nn
import networkx as nx
import itertools

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator

# --------------------------------------------------------------------------- #
#  Quantum helpers
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit data‑encoding and variational
    parameters. The encoding consists of RX rotations and the variational
    part alternates between RY rotations and a fixed CZ entangling layer.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for q, param in enumerate(encoding):
        circuit.rx(param, q)

    w_idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[w_idx], q)
            w_idx += 1
        # CZ entanglement across nearest neighbours
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
#  Synthetic data utilities (adapted from GraphQNN)
# --------------------------------------------------------------------------- #
def random_training_data(unitary: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate a training set where each input state is a random pure state
    and the target is the state after applying *unitary*.
    """
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        amp = torch.randn(dim, dtype=torch.complex64)
        amp /= torch.norm(amp)
        state = amp
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Build a random sequence of unitary layers that map a *qnn_arch[0]* qubit
    input to a *qnn_arch[-1]* qubit output. The final unitary serves as the
    target for the synthetic training set.
    """
    target_unitary = torch.linalg.qr(torch.randn(2 ** qnn_arch[-1], 2 ** qnn_arch[-1], dtype=torch.complex64))[0]
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[torch.Tensor]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[torch.Tensor] = []
        for _ in range(num_outputs):
            op = torch.linalg.qr(torch.randn(2 ** (num_inputs + 1), 2 ** (num_inputs + 1), dtype=torch.complex64))[0]
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace(state: torch.Tensor, keep: List[int]) -> torch.Tensor:
    """
    Very small helper that keeps the computational basis indices
    specified by *keep*.  This is a mock implementation that simply
    slices the state vector; it is sufficient for synthetic data.
    """
    size = state.shape[0]
    mask = torch.zeros(size, dtype=torch.bool)
    mask[keep] = True
    return state[mask]


def feedforward(
    qnn_arch: List[int],
    unitaries: List[List[torch.Tensor]],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """
    Forward‑propagate a batch of states through the layered unitaries.
    Returns a list of state lists for each sample.
    """
    stored: List[List[torch.Tensor]] = []
    for state, _ in samples:
        layerwise: List[torch.Tensor] = [state]
        current = state
        for layer in range(1, len(qnn_arch)):
            # prepend an auxiliary zero state for the extra qubit
            zero = torch.zeros(2 ** (qnn_arch[layer] - 1), dtype=torch.complex64)
            current = torch.cat([current, zero])
            # apply the first unitary of this layer
            current = unitaries[layer][0] @ current
            # apply the remaining unitaries
            for gate in unitaries[layer][1:]:
                current = gate @ current
            # partial trace out the auxiliary qubits
            keep = list(range(2 ** qnn_arch[layer]))
            current = _partial_trace(current, keep)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the absolute squared overlap of two pure states."""
    return abs(torch.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: List[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Create a weighted adjacency graph from state fidelities.
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


# --------------------------------------------------------------------------- #
#  Quantum classifier wrapper
# --------------------------------------------------------------------------- #
class QuantumClassifier(nn.Module):
    """
    Run the parameterised circuit on a qiskit simulator and return the
    expectation values of the Pauli‑Z observables.
    """
    def __init__(self, num_qubits: int, depth: int, backend=AerSimulator()):
        super().__init__()
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = backend

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        *inputs* must be a tensor of shape (batch, num_qubits) containing
        real numbers that will be used to instantiate the encoding parameters.
        """
        batch_size = inputs.shape[0]
        # bind encoding parameters
        bound_circuits = []
        for i in range(batch_size):
            bound = self.circuit.bind_parameters(
                {p: inputs[i, idx].item() for idx, p in enumerate(self.encoding)}
            )
            bound_circuits.append(bound)

        # transpile and run
        transpiled = transpile(bound_circuits, self.backend)
        job = self.backend.run(transpiled)
        result = job.result()

        # gather expectation values
        expectations = []
        for idx, circ in enumerate(transpiled):
            exp = []
            for obs in self.observables:
                exp.append(result.get_expectation_value(obs, circ))
            expectations.append(exp)

        return torch.tensor(expectations, dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Hybrid wrapper
# --------------------------------------------------------------------------- #
class UnifiedQuantumClassifier(nn.Module):
    """
    Hybrid model that first runs the quantum circuit to obtain a feature
    vector and then feeds those features into a classical PyTorch classifier.
    The classical classifier is built with *build_classifier_circuit* from
    the ML module, but here we construct it directly for convenience.
    """
    def __init__(self, num_qubits: int, depth: int, classical_depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.quantum = QuantumClassifier(num_qubits, depth)
        # The output dimension of the quantum part equals the number of qubits
        self.classifier, _, _, _ = build_classifier_circuit(
            num_qubits, classical_depth, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantum_features = self.quantum(x)
        return self.classifier(quantum_features)


__all__ = [
    "build_classifier_circuit",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "QuantumClassifier",
    "UnifiedQuantumClassifier",
]
