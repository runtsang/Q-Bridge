"""Quantum estimator based on a simple variational circuit and graph utilities."""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

__all__ = [
    "EstimatorQNN",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]

# ----- Utility functions inspired by GraphQNN ---------------------------------

def _tensored_id(num_qubits: int):
    """Return an identity operator on ``num_qubits`` qubits."""
    return qiskit.quantum_info.Operator(np.eye(2**num_qubits))

def _tensored_zero(num_qubits: int):
    """Return the zero projector on ``num_qubits`` qubits."""
    zero = np.zeros((2**num_qubits, 1))
    zero[0, 0] = 1.0
    return qiskit.quantum_info.Operator(zero)

def _swap_registers(op: qiskit.quantum_info.Operator, source: int, target: int):
    """Swap registers ``source`` and ``target`` in an operator."""
    if source == target:
        return op
    perm = list(range(op.shape[0]))
    perm[source], perm[target] = perm[target], perm[source]
    return op.permute(perm)

def _random_qubit_unitary(num_qubits: int):
    """Generate a random unitary on ``num_qubits`` qubits."""
    dim = 2**num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return qiskit.quantum_info.Operator(q)

def _random_qubit_state(num_qubits: int):
    """Generate a random pure state on ``num_qubits`` qubits."""
    dim = 2**num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return qiskit.quantum_info.Operator(vec.reshape(-1, 1))

def random_training_data(unitary: qiskit.quantum_info.Operator, samples: int):
    """Create input–output pairs for a given unitary."""
    dataset = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    """
    Generate a random layered unitary network and training data.

    Parameters
    ----------
    qnn_arch : list[int]
        Architecture: number of qubits per layer.
    samples : int
        Number of training samples.

    Returns
    -------
    tuple[list[int], list[list[Operator]], list[tuple[Operator, Operator]], Operator]
        Architecture, list of lists of unitary layers, training data, target unitary.
    """
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qiskit.quantum_info.Operator]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qiskit.quantum_info.Operator] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qiskit.quantum_info.Operator(
                    np.kron(_random_qubit_unitary(num_inputs + 1).data, np.eye(2**(num_outputs - 1)))
                )
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qiskit.quantum_info.Operator, keep: list[int]):
    """Partial trace over qubits not in ``keep``."""
    if len(keep) == len(state.shape):
        return state
    return state.ptrace(keep)

def _partial_trace_remove(state: qiskit.quantum_info.Operator, remove: list[int]):
    """Partial trace over qubits in ``remove``."""
    keep = [i for i in range(state.shape[0]) if i not in remove]
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: list[int],
    unitaries: list[list[qiskit.quantum_info.Operator]],
    layer: int,
    input_state: qiskit.quantum_info.Operator,
):
    """Apply one layer of the network to an input state."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qiskit.quantum_info.Operator(
        np.kron(input_state.data, np.zeros((2**num_outputs, 1))).reshape(-1, 1)
    )
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary
    return _partial_trace_remove(layer_unitary @ state @ layer_unitary.adjoint(), list(range(num_inputs)))

def feedforward(
    qnn_arch: list[int],
    unitaries: list[list[qiskit.quantum_info.Operator]],
    samples: list[tuple[qiskit.quantum_info.Operator, qiskit.quantum_info.Operator]],
):
    """Run the network on a batch of samples and record intermediate states."""
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qiskit.quantum_info.Operator, b: qiskit.quantum_info.Operator) -> float:
    """Absolute squared overlap between pure states."""
    return abs((a.adjoint() @ b)[0, 0]) ** 2

def fidelity_adjacency(
    states: list[qiskit.quantum_info.Operator],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
):
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

# ----- Quantum estimator -------------------------------------------------------

class EstimatorQNN:
    """
    Variational quantum estimator that accepts a 1‑D input parameter and a weight parameter.
    The circuit is a single qubit with an H gate followed by a rotation about Y (input)
    and a rotation about Z (weight).  The expectation value of the Y Pauli operator
    is returned as the prediction.
    """

    def __init__(self, backend="qiskit.aer.noise.NoiseModel"):
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")

        # Circuit construction
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rz(self.weight_param, 0)

        # Observable
        observable = SparsePauliOp.from_list([("Y", 1)])

        # Estimator wrapper
        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=estimator,
        )

    def evaluate(self, input_value: float, weight_value: float) -> float:
        """
        Compute the expectation value for a given input and weight.

        Parameters
        ----------
        input_value : float
            Value to bind to the input parameter.
        weight_value : float
            Value to bind to the weight parameter.

        Returns
        -------
        float
            Expected value of the Y operator.
        """
        return float(
            self.estimator_qnn.evaluate(
                input_params={self.input_param: input_value},
                weight_params={self.weight_param: weight_value},
            )[0]
        )
