"""GraphQNNHybrid – quantum implementation using qiskit and QCNN ansatz.

The class provides the same public interface as the classical
implementation but replaces the convolutional filter with a
parameterized 2×2 quantum circuit and the GNN core with a QCNN‑style
ansatz.  All operations are performed on statevectors; training data
consists of pairs of input states and target states produced by a
random unitary.

Key components
---------------
* `QuanvCircuit` – 2×2 quantum filter that encodes a 2×2 image patch.
* `QCNNAnsatz` – QCNN‑style layered ansatz for message passing.
* `fidelity_adjacency` – builds a weighted graph from state fidelities.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator

Tensor = np.ndarray  # classical numpy array representing a state vector

# --------------------------------------------------------------------------- #
#  Quantum filter – 2×2 patch encoding
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Quantum filter that processes a 2×2 image patch.

    The circuit applies an RX rotation to each qubit based on the pixel
    intensity, followed by a random two‑qubit circuit, and measures
    all qubits.  The average probability of measuring |1> is returned.
    """
    def __init__(self, kernel_size: int = 2, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        # add a small random circuit to increase expressivity
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, patch: np.ndarray) -> float:
        """Execute the filter on a single 2×2 patch."""
        # map pixel intensities to rotation angles
        data = patch.reshape(1, self.n_qubits)
        binds = []
        for d in data:
            bind = {theta: np.pi if val > self.threshold else 0 for val, theta in zip(d, self.theta)}
            binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        # compute average number of |1> across all shots
        total_ones = sum(int(bit) * count for bit, count in counts.items() for _ in range(len(bit)))
        return total_ones / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
#  QCNN‑style ansatz – layered circuit for message passing
# --------------------------------------------------------------------------- #
def _conv_circuit(params: Sequence[float]) -> QuantumCircuit:
    """Two‑qubit convolution unit defined in the QCNN reference."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def conv_layer(num_qubits: int, param_prefix: str, param_values: Sequence[float]) -> QuantumCircuit:
    """Construct a convolutional layer that applies _conv_circuit to each pair."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        params = param_values[param_index:param_index+3]
        sub = _conv_circuit(params)
        qc.append(sub, [q1, q2])
        param_index += 3
    return qc

def pool_layer(num_qubits: int, param_prefix: str, param_values: Sequence[float]) -> QuantumCircuit:
    """Pooling layer that measures and discards qubits."""
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        params = param_values[param_index:param_index+3]
        sub = _conv_circuit(params)
        qc.append(sub, [q1, q2])
        param_index += 3
    return qc

class QCNNAnsatz:
    """QCNN‑style ansatz built from alternating convolution and pooling layers."""
    def __init__(self, layers: int = 3, base_qubits: int = 8):
        self.layers = layers
        self.base_qubits = base_qubits
        self.circuit = QuantumCircuit(base_qubits)
        # random parameter vector for all layers
        self.params = np.random.randn(self.layers * (self.base_qubits // 2) * 3)

        # build the layered circuit
        idx = 0
        for l in range(self.layers):
            # convolution
            conv_params = self.params[idx:idx + (self.base_qubits // 2) * 3]
            self.circuit.append(conv_layer(self.base_qubits, f"c{l}", conv_params), range(self.base_qubits))
            idx += (self.base_qubits // 2) * 3
            # pooling – reduce qubit count by half
            pool_params = self.params[idx:idx + (self.base_qubits // 4) * 3]
            self.circuit.append(pool_layer(self.base_qubits, f"p{l}", pool_params), range(self.base_qubits))
            idx += (self.base_qubits // 4) * 3
            # shrink the active register
            self.base_qubits //= 2
        # final measurement
        self.circuit.measure_all()

    def run(self, state: Statevector) -> Statevector:
        """Apply the ansatz to a statevector and return the resulting state."""
        job = execute(self.circuit, AerSimulator(method="statevector"))
        result = job.result()
        return Statevector(result.get_statevector())

# --------------------------------------------------------------------------- #
#  GraphQNNHybrid – quantum class
# --------------------------------------------------------------------------- #
class GraphQNNHybrid:
    """Hybrid quantum graph neural network.

    The constructor mirrors the classical API but internally uses
    quantum circuits for the convolutional front‑end and the GNN core.
    """
    def __init__(self, arch: Sequence[int], use_conv: bool = True, conv_kernel: int = 2, conv_threshold: float = 0.5):
        self.arch = list(arch)
        self.use_conv = use_conv
        self.conv = QuanvCircuit(kernel_size=conv_kernel, threshold=conv_threshold)
        self.ansatz = QCNNAnsatz(layers=len(arch)-1, base_qubits=arch[0])

    def forward(self, data: Tensor, graph: nx.Graph) -> List[Statevector]:
        """
        Parameters
        ----------
        data : Tensor
            Input image array of shape (B, H, W).
        graph : nx.Graph
            Graph over the batch.  Each node corresponds to a sample.
        Returns
        -------
        activations : List[Statevector]
            List of statevectors after each layer of the ansatz.
        """
        # 1. Convolutional filtering – produce a classical scalar per patch
        batch_features = []
        for img in data:
            # split into 2×2 patches
            patches = [img[i:i+2, j:j+2] for i in range(0, img.shape[0], 2) for j in range(0, img.shape[1], 2)]
            vals = [self.conv.run(p) for p in patches]
            batch_features.append(np.mean(vals))
        batch_features = np.array(batch_features)

        # 2. Encode features into a statevector (amplitude encoding)
        states = [Statevector.from_label(bin(int(f))[2:].zfill(1)) for f in batch_features]

        # 3. Apply QCNN ansatz layer‑wise
        activations: List[Statevector] = [states[0]]  # placeholder for first sample
        for i, state in enumerate(states):
            new_state = self.ansatz.run(state)
            activations.append(new_state)
        return activations

# --------------------------------------------------------------------------- #
#  Utility functions – quantum analogues
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(n_qubits: int) -> Statevector:
    dim = 2 ** n_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    mat = np.linalg.qr(mat)[0]  # orthonormal
    return Statevector(mat)

def _random_qubit_state(n_qubits: int) -> Statevector:
    dim = 2 ** n_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)

def random_training_data(unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    dataset = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.num_qubits)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    # For simplicity, we return the unitary as the “weights” of the network
    return list(qnn_arch), [target_unitary], training_data, target_unitary

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Statevector],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    stored: List[List[Statevector]] = []
    for state, _ in samples:
        activations = [state]
        current = state
        for unitary in unitaries:
            current = unitary @ current
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Statevector, b: Statevector) -> float:
    return abs((a.data.conj().T @ b.data)[0, 0]) ** 2

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

__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
