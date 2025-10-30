"""Hybrid quantum GraphQNN incorporating self‑attention style parameterisation,
a quantum estimator, and a quanvolution filter.

The implementation is intentionally lightweight – all heavy lifting
is delegated to Qiskit primitives, so the code remains concise and
easily extensible.  The public API mirrors the classical variant
and can be swapped in a single line of code.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator
import qutip as qt

# --------------------------------------------------------------------------- #
# Basic utilities – mirror the classical helpers but with Qobj
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    return qt.Qobj(unitary, dims=[[2] * num_qubits, [2] * num_qubits])

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amp = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amp /= np.linalg.norm(amp)
    return qt.Qobj(amp, dims=[[2] * num_qubits, [1] * num_qubits])

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(n)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data  = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_in = qnn_arch[layer - 1]
        num_out = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for _ in range(num_out):
            op = _random_qubit_unitary(num_in + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_in = qnn_arch[layer - 1]
    num_out = qnn_arch[layer]
    state = qt.tensor(input_state, qt.qeye(2 ** num_out))
    op = qt.tensor(unitaries[layer][0], qt.qeye(2 ** (num_out - 1)))  # simple placeholder
    out = op * state * op.dag()
    return qt.ptrace(out, [i for i in range(num_in)])

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    results: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        results.append(layerwise)
    return results

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
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
# Sub‑module 1 – quantum self‑attention
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Self‑attention block implemented with a Qiskit circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(entangle_params[i], i + 1)
        return qc

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = backend.run(qc, shots=shots)
        return job.result().get_counts(qc)

# --------------------------------------------------------------------------- #
# Sub‑module 2 – quantum estimator
# --------------------------------------------------------------------------- #
def _quantum_estimator():
    """Builds a lightweight Qiskit EstimatorQNN."""
    params = [qiskit.circuit.Parameter("theta")]
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.rx(params[0], 0)
    observable = Pauli("Z")
    estimator = QiskitEstimator()
    return QiskitEstimatorQNN(circuit=circuit,
                              observables=[observable],
                              input_params=[],
                              weight_params=[params[0]],
                              estimator=estimator)

# --------------------------------------------------------------------------- #
# Sub‑module 3 – quanvolution filter
# --------------------------------------------------------------------------- #
class QuantumQuanvolutionFilter:
    """Applies a small quantum kernel to 2×2 patches of a 28×28 image."""
    def __init__(self, n_wires: int = 4):
        self.n_wires = n_wires
        self.backend = qiskit.Aer.get_backend("statevector_simulator")

    def _patch_circuit(self, patch: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_wires)
        for i, val in enumerate(patch.flatten()):
            qc.ry(val, i)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        return qc

    def run(self, image: np.ndarray) -> np.ndarray:
        """Return a 1‑D feature vector for the whole image."""
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = image[:, r, c:r+2, c+2:c+4]
                qc = self._patch_circuit(patch)
                result = self.backend.run(qc).result()
                vec = result.get_statevector()
                patches.append(vec)
        return np.concatenate(patches, axis=1)

# --------------------------------------------------------------------------- #
# Main hybrid class – quantum version
# --------------------------------------------------------------------------- #
class HybridGraphQNN:
    """
    Quantum counterpart of the classical ``HybridGraphQNN``.
    Operates on qiskit Statevector representations and uses
    variational circuits for both graph propagation and self‑attention.
    """
    def __init__(
        self,
        graph_arch: Sequence[int],
        use_self_attention: bool = True,
        use_estimator: bool = True,
        use_quanvolution: bool = False,
    ) -> None:
        self.graph_arch = list(graph_arch)
        self.use_self_attention = use_self_attention
        self.use_estimator      = use_estimator
        self.use_quanvolution   = use_quanvolution

        # Build quantum sub‑modules
        if use_self_attention:
            self.attention = QuantumSelfAttention(n_qubits=len(graph_arch))
        if use_estimator:
            self.estimator = _quantum_estimator()
        if use_quanvolution:
            self.qfilter = QuantumQuanvolutionFilter()

        # Random quantum graph network
        _, self.unitaries, _, _ = random_network(self.graph_arch, samples=100)

        # Aer backend for all simulation
        self.backend = qiskit.Aer.get_backend("statevector_simulator")

    # --------------------------------------------------------------------- #
    # Graph mode
    # --------------------------------------------------------------------- #
    def _graph_forward(self, state: qt.Qobj, graph: nx.Graph | None = None) -> qt.Qobj:
        if graph is None:
            graph = nx.Graph()
            graph.add_nodes_from(range(1))

        # Self‑attention – use the same parameters for all nodes
        if self.use_self_attention:
            rot = np.random.uniform(0, 2 * np.pi, 3 * len(self.graph_arch))
            ent = np.random.uniform(0, 2 * np.pi, len(self.graph_arch) - 1)
            attn_counts = self.attention.run(self.backend, rot, ent)
            # Convert measurement histogram to a state (placeholder)
            state = qt.Qobj(np.array([1.0]), dims=[[2], [1]])

        # Feed‑forward through the quantum network
        samples = [(state, None)]
        activations = feedforward(self.graph_arch, self.unitaries, samples)
        final = activations[-1][-1]

        if self.use_estimator:
            # Evaluate estimator
            result = self.estimator.run(self.backend, weights=[0.5], inputs=final)
            return result
        return final

    # --------------------------------------------------------------------- #
    # Quanvolution mode
    # --------------------------------------------------------------------- #
    def _quanvolution_forward(self, image: np.ndarray) -> np.ndarray:
        if self.use_quanvolution:
            return self.qfilter.run(image)
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Public forward
    # --------------------------------------------------------------------- #
    def forward(self, input_data, graph: nx.Graph | None = None):
        """
        Dispatches to the appropriate quantum branch.
        ``input_data`` can be a Statevector or a NumPy image array.
        """
        if self.use_quanvolution and isinstance(input_data, np.ndarray) and input_data.shape[-2:] == (28, 28):
            return self._quanvolution_forward(input_data)
        if isinstance(input_data, qt.Qobj):
            return self._graph_forward(input_data, graph)
        raise TypeError("Unsupported input type for HybridGraphQNN quantum branch.")

__all__ = [
    "HybridGraphQNN",
    "QuantumSelfAttention",
    "QuantumQuanvolutionFilter",
    "_quantum_estimator",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
