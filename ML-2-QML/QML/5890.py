"""Quantum‑only utilities for the hybrid GraphQNN.

Provides a parameterised quantum circuit generator, training data
generation, a feed‑forward routine that returns state vectors, and
fidelity‑based graph construction.  The implementation is a direct
adaptation of the original QML seed but updated to use the
parameter‑shift rule for gradients and to expose a convenient
EstimatorQNN wrapper that can be mixed with the classical module.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli
from qiskit.primitives import Estimator as QiskitEstimator

# --------------------------------------------------------------------------- #
# Quantum utilities – adapted from the original GraphQNN QML seed
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> QuantumCircuit:
    """Return a random unitary circuit for *num_qubits* qubits."""
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.rx(np.random.randn(), i)
        qc.ry(np.random.randn(), i)
        qc.rz(np.random.randn(), i)
    return qc

def random_training_data(unitary: QuantumCircuit, samples: int) -> List[Tuple[QuantumCircuit, QuantumCircuit]]:
    """Generate pairs of input states and their transformed counterparts."""
    data: List[Tuple[QuantumCircuit, QuantumCircuit]] = []
    for _ in range(samples):
        # random input state |+> on each qubit
        input_qc = QuantumCircuit(unitary.num_qubits)
        for q in range(unitary.num_qubits):
            input_qc.h(q)
        # target state = unitary * |+>
        target_qc = unitary.copy()
        target_qc.compose(input_qc, inplace=True)
        data.append((input_qc, target_qc))
    return data

def random_network(arch: Sequence[int], samples: int):
    """Generate a random unitary per layer and training data."""
    target_unitary = _random_qubit_unitary(arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return list(arch), training_data, target_unitary

def feedforward(
    arch: Sequence[int],
    unitaries: Sequence[QuantumCircuit],
    samples: Iterable[Tuple[QuantumCircuit, QuantumCircuit]],
) -> List[List[QuantumCircuit]]:
    """Propagate input states through the unitary layers."""
    outputs: List[List[QuantumCircuit]] = []
    for inp, _ in samples:
        layerwise = [inp]
        current = inp.copy()
        for layer_unitary in unitaries:
            current.compose(layer_unitary, inplace=True)
            layerwise.append(current.copy())
        outputs.append(layerwise)
    return outputs

def state_fidelity(a: QuantumCircuit, b: QuantumCircuit) -> float:
    """Fidelity between the output states of two circuits."""
    backend = Aer.get_backend("statevector_simulator")
    vec_a = Statevector.from_instruction(a).data
    vec_b = Statevector.from_instruction(b).data
    return np.abs(np.vdot(vec_a, vec_b)) ** 2

def fidelity_adjacency(
    states: Sequence[QuantumCircuit],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, aj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# Quantum Estimator wrapper – compatible with the classical hybrid
# --------------------------------------------------------------------------- #

class GraphQNNHybrid:
    """
    Thin wrapper around Qiskit’s Estimator that exposes the same
    EstimatorQNN interface used in the classical module.

    The circuit consists of a single qubit with a tunable Ry gate.
    The observable is Pauli‑Y.  The wrapper supports the
    parameter‑shift rule for gradient computation.
    """

    def __init__(self) -> None:
        self.observable = Pauli.from_label("Y")
        self.simulator = Aer.get_backend("statevector_simulator")
        self.estimator = QiskitEstimator(self.simulator)

    def __call__(self, angle: float) -> float:
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(angle, 0)
        result = self.estimator.run(qc, observables=[self.observable])
        exp_val = result.result()[0].real
        return exp_val

    def gradient(self, angle: float, shift: float = np.pi / 2) -> float:
        """Compute the gradient via the parameter‑shift rule."""
        f_plus = self.__call__(angle + shift)
        f_minus = self.__call__(angle - shift)
        return 0.5 * (f_plus - f_minus)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
