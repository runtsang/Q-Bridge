"""
GraphQNNGen448: quantum counterpart.

The module mirrors the classical interface but uses Qiskit circuits,
state‑vector simulations and quantum fidelity calculations.
It also provides a quantum fraud‑detection circuit, a quantum
classifier ansatz, and a quanvolution filter.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

Tensor = np.ndarray


# --------------------------------------------------------------------------- #
# 1.  Fraud‑detection parameters and helper
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters that describe a single photonic fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a Qiskit program that mimics the photonic fraud‑detection circuit."""
    prog = QuantumCircuit(2)
    # Helper that encodes a single photonic layer using RX, RZ, etc.
    def _apply_layer(circ: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
        circ.rx(params.bs_theta, 0)
        circ.rx(params.bs_phi, 1)
        for i, phase in enumerate(params.phases):
            circ.rz(phase, i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            circ.ry(_clip(r, 5.0) if clip else r, i)
        circ.rx(params.bs_theta, 0)
        circ.rx(params.bs_phi, 1)
        for i, phase in enumerate(params.phases):
            circ.rz(phase, i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            circ.ry(_clip(r, 5.0) if clip else r, i)
        for i, k in enumerate(params.kerr):
            circ.ry(_clip(k, 1.0) if clip else k, i)

    _apply_layer(prog, input_params, clip=False)
    for lay in layers:
        _apply_layer(prog, lay, clip=True)
    return prog


# --------------------------------------------------------------------------- #
# 2.  Quanvolution filter
# --------------------------------------------------------------------------- #
def Conv() -> QuantumCircuit:
    """Return a simple quanvolution filter circuit."""
    filter_size = 2
    n_qubits = filter_size ** 2
    qc = QuantumCircuit(n_qubits)
    theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        qc.rx(theta[i], i)
    qc.barrier()
    qc += qiskit.circuit.random.random_circuit(n_qubits, 2, seed=42)
    qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# 3.  Quantum graph neural network utilities
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> Statevector:
    """Generate a random unitary as a state‑vector by exponentiating a random matrix."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    u, _, vh = np.linalg.svd(matrix, full_matrices=False)
    return Statevector(u, dims=[2] * num_qubits)


def random_network(qnn_arch: List[int], samples: int):
    """Generate a random quantum network and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    # Build a list of parameter vectors per layer
    unitaries: List[List[ParameterVector]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_params: List[ParameterVector] = []
        for _ in range(num_outputs):
            pv = ParameterVector(f"theta_{layer}", num_inputs + 1)
            layer_params.append(pv)
        unitaries.append(layer_params)
    return qnn_arch, unitaries, training_data, target_unitary


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[ParameterVector]],
    layer: int,
    input_state: Statevector,
) -> Statevector:
    """Apply a single quantum layer and trace out the inputs."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Prepare a composite state by appending |0>^{num_outputs}
    ancilla = Statevector.from_label("0" * num_outputs, dims=[2] * num_outputs)
    state = input_state.tensor(ancilla)

    # Build the unitary for this layer
    circ = QuantumCircuit(num_inputs + num_outputs)
    for pv in unitaries[layer]:
        for gate, param in zip([circ.rx, circ.rz], pv):
            circ.append(gate(param), gate.qubits)
    # Apply the unitary
    circ = circ.to_instruction()
    new_state = state.evolve(circ)
    # Trace out the input qubits
    keep = list(range(num_outputs))
    return new_state.truncate(keep)


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[ParameterVector]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Propagate a batch of quantum states through the network."""
    stored: List[List[Statevector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored


# --------------------------------------------------------------------------- #
# 4.  Fidelity helpers
# --------------------------------------------------------------------------- #
def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return the squared overlap of two pure statevectors."""
    return abs(np.vdot(a.data, b.data)) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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
# 5.  Quantum classifier factory
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter], List[SparsePauliOp]]:
    """Create a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, q in zip(encoding, range(num_qubits)):
        qc.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return qc, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# 6.  Training data generation for quantum networks
# --------------------------------------------------------------------------- #
def random_training_data(unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate input‑output pairs for a quantum network."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = Statevector.from_label("0" * num_qubits, dims=unitary.dims)
        output = state.evolve(unitary.to_instruction())
        dataset.append((state, output))
    return dataset


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "Conv",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "build_classifier_circuit",
    "random_training_data",
]
