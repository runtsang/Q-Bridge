from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
import networkx as nx
import numpy as np

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator as QiskitEstimator

def random_parameters(num_qubits: int, num_layers: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_layers, 2 * num_qubits))

def build_circuit(num_qubits: int, parameters: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    num_layers, num_params = parameters.shape
    assert num_params == 2 * num_qubits, "Parameter shape mismatch"
    for layer in range(num_layers):
        for q in range(num_qubits):
            qc.ry(parameters[layer, q], q)
            qc.rz(parameters[layer, q + num_qubits], q)
        # nearest‑neighbour entanglement
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    return qc

def state_fidelity(a: Statevector, b: Statevector) -> float:
    return float(abs(np.vdot(a.data, b.data)) ** 2)

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, s1), (j, s2) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s1, s2)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

class UnifiedEstimatorQNN:
    """
    Variational quantum circuit that mirrors the classical architecture.
    Each layer contains parametrized Ry/Rz rotations on all qubits followed by
    a nearest‑neighbour CNOT chain.  The first layer is conditioned on the
    classical input, providing a hybrid encoder.  The circuit is evaluated
    with a simple Pauli‑Z observable and the expectation value serves as the
    output of the network.
    """
    def __init__(
        self,
        arch: Sequence[int],
        num_layers: int = 2,
        seed: int | None = None,
        fidelity_threshold: float = 0.9,
    ) -> None:
        self.arch = list(arch)
        self.num_qubits = arch[0]
        self.num_layers = num_layers
        self.fidelity_threshold = fidelity_threshold
        self.parameters = random_parameters(self.num_qubits, self.num_layers, seed)
        self.estimator = QiskitEstimator()
        self.observable = SparsePauliOp.from_list([("Z" * self.num_qubits, 1)])

    def _build_circuit(self, input_angles: np.ndarray | None = None) -> QuantumCircuit:
        params = self.parameters.copy()
        if input_angles is not None:
            params[0, : self.num_qubits] = input_angles
        return build_circuit(self.num_qubits, params)

    def forward(self, input_angles: np.ndarray) -> float:
        qc = self._build_circuit(input_angles)
        result = self.estimator.run(qc, observables=self.observable).result()
        return result.values[0]

    def evaluate_states(self, inputs: Iterable[np.ndarray]) -> list[Statevector]:
        states = []
        for inp in inputs:
            qc = self._build_circuit(inp)
            sv = Statevector.from_instruction(qc)
            states.append(sv)
        return states

    def fidelity_graph(self, states: Sequence[Statevector]) -> nx.Graph:
        return fidelity_adjacency(states, self.fidelity_threshold)

    def __repr__(self) -> str:
        return f"<UnifiedEstimatorQNN qubits={self.num_qubits} layers={self.num_layers} params={self.parameters.size}>"

__all__ = [
    "UnifiedEstimatorQNN",
    "random_parameters",
    "build_circuit",
    "state_fidelity",
    "fidelity_adjacency",
]
