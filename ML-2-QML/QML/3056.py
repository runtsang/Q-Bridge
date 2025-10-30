"""GraphQNNHybrid: Quantum‑centric graph neural network utilities.

This module mirrors the classical counterpart but uses Qiskit‑based
variational circuits, state‑vector sampling, and a quantum auto‑encoder.
It retains the same public API while adding a sampler‑based forward
pass and a hybrid auto‑encoder that can be trained on a quantum back‑end.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------- #
# Quantum utilities (mirrored from QML‑GraphQNN.py)
# --------------------------------------------------------------------------- #

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


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Return architecture, unitaries, training data and target unitary."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
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

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Quantum forward pass using partial traces after each layer."""
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise: List[qt.Qobj] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: Optional[float] = None,
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
# Quantum auto‑encoder (mirrored from QML‑Autoencoder.py)
# --------------------------------------------------------------------------- #

def Autoencoder() -> SamplerQNN:
    """Return a Qiskit SamplerQNN auto‑encoder circuit."""
    sampler = Sampler()

    def ansatz(num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        auxiliary_qubit = num_latent + 2 * num_trash
        # Swap test
        circuit.h(auxiliary_qubit)
        for i in range(num_trash):
            circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary_qubit)
        circuit.measure(auxiliary_qubit, cr[0])
        return circuit

    num_latent = 3
    num_trash = 2
    qc = auto_encoder_circuit(num_latent, num_trash)

    # Interpret returns the measurement probability as a float
    def identity_interpret(x: List[float]) -> List[float]:
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


# --------------------------------------------------------------------------- #
# Hybrid training helpers (quantum‑classical)
# --------------------------------------------------------------------------- #

def hybrid_train_q(
    qnn_arch: Sequence[int],
    autoencoder_qnn: SamplerQNN,
    data: List[qt.Qobj],
    *,
    epochs: int = 10,
    shots: int = 1024,
) -> List[float]:
    """
    Jointly train a quantum QNN and a quantum auto‑encoder.
    The loss is a weighted sum of the QNN regression loss (state fidelity)
    and the auto‑encoder reconstruction loss (classical MSE on probabilities).
    """
    # Build quantum QNN unitaries
    weights = [_random_qubit_unitary(num) for num in qnn_arch[1:]]
    # Optimizer placeholder (requires a quantum compatible optimizer, e.g. COBYLA)
    # For brevity, we only illustrate a single forward step.
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for state in data:
            # QNN forward
            current = state
            for w in weights:
                current = w * current
            qnn_loss = 1 - state_fidelity(current, state)

            # AutoEncoder forward
            probs = autoencoder_qnn.eval(state, shots=shots)
            recon = qt.Qobj(probs[0], dims=[[2], [1]])  # simplistic mapping
            ae_loss = 1 - state_fidelity(recon, state)

            loss = 0.6 * qnn_loss + 0.4 * ae_loss
            epoch_loss += loss
        epoch_loss /= len(data)
        history.append(epoch_loss)
    return history


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "Autoencoder",
    "hybrid_train_q",
]
