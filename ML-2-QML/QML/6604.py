"""GraphQNNHybrid – quantum implementation mirroring the classical API.

The quantum version uses a variational RealAmplitudes ansatz to encode
the input features, followed by a swap‑test style block that extracts a
latent quantum state.  The subsequent layers are constructed from
random unitaries acting on the latent register, and a partial trace
keeps only the output qubits.  Fidelity‑based adjacency is computed
using statevector overlaps, identical to the classical case.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import qiskit as qk
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.primitives import StatevectorSampler as Sampler


# --------------------------------------------------------------------------- #
# Helper functions – minimal quantum equivalents of the classical utilities
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> qk.quantum_info.QubitStateVector:
    """Generate a random unitary matrix acting on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    qobj = qk.quantum_info.QubitStateVector(matrix)
    return qobj


def _tensored_zero(num_qubits: int) -> qk.quantum_info.QubitStateVector:
    """Projector onto |0⟩ for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1.0
    return qk.quantum_info.QubitStateVector(vec)


def _partial_trace(state: Statevector, keep: Sequence[int]) -> Statevector:
    """Return the reduced state over the qubits in ``keep``."""
    return state.reduce(keep)


# --------------------------------------------------------------------------- #
# Quantum autoencoder – analogous to the classical Autoencoder
# --------------------------------------------------------------------------- #
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Construct a quantum circuit that encodes ``num_latent`` logical qubits
    and uses a swap‑test style block to produce a latent state.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Feature encoding – RealAmplitudes ansatz
    circuit.compose(RealAmplitudes(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    # Auxiliary qubit for swap‑test
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])
    return circuit


# --------------------------------------------------------------------------- #
# Graph‑neural‑network core – quantum variant
# --------------------------------------------------------------------------- #
class GraphQNNHybridQML:
    """
    Quantum analogue of :class:`GraphQNNHybrid`.  Public methods mirror the
    classical API, enabling zero‑copy switching between regimes.
    """

    def __init__(self, arch: Sequence[int], num_trash: int = 2) -> None:
        self.arch = list(arch)
        self.num_trash = num_trash
        self.sampler = Sampler()
        self.unitaries: List[List[Statevector]] = [[]]

    def random_network(self, samples: int) -> Tuple[List[int], List[List[Statevector]], List[Tuple[Statevector, Statevector]], Statevector]:
        """Generate random unitaries and training data for the quantum network."""
        # Target unitary is the last layer’s unitary
        target_unitary = _random_qubit_unitary(self.arch[-1])
        training_data = random_training_data(target_unitary, samples)

        # Random layer unitaries
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[Statevector] = []
            for output in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    # tensor with identity for remaining outputs
                    op = op.tensor(_tensored_zero(num_outputs - 1))
                    op = op.swap(num_inputs, num_inputs + output)
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

        return self.arch, self.unitaries, training_data, target_unitary

    def feedforward(
        self,
        samples: Iterable[Tuple[Statevector, Statevector]],
    ) -> List[List[Statevector]]:
        """Propagate each sample through the quantum layers."""
        stored_states: List[List[Statevector]] = []
        for state, _ in samples:
            layerwise = [state]
            current = state
            for layer in range(1, len(self.arch)):
                current = self._layer_channel(layer, current)
                layerwise.append(current)
            stored_states.append(layerwise)
        return stored_states

    def _layer_channel(self, layer: int, input_state: Statevector) -> Statevector:
        """Apply the layer’s unitaries and perform a partial trace."""
        num_inputs = self.arch[layer - 1]
        num_outputs = self.arch[layer]
        # Pad input state with |0⟩ for the output qubits
        padded = input_state.tensor(_tensored_zero(num_outputs))
        # Compose unitaries
        op = self.unitaries[layer][0]
        for gate in self.unitaries[layer][1:]:
            op = op.compose(gate)
        # Apply unitary and trace out the input register
        out_state = op.apply(padded)
        return _partial_trace(out_state, list(range(num_inputs)))

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Squared magnitude of the inner product of two pure states."""
        return abs((a.data.conj().T @ b.data)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from quantum state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybridQML.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # --------------------------------------------------------------------- #
    # Simple training routine – gradient‑free optimisation of the unitary
    # --------------------------------------------------------------------- #
    def train(
        self,
        samples: List[Tuple[Statevector, Statevector]],
        epochs: int = 20,
        optimizer_cls: type = COBYLA,
    ) -> List[float]:
        """
        Optimize the parameters of the last unitary to minimise the
        mean‑squared‑error between the network output and the target
        state.  Only the parameters of the final layer are updated.
        """
        opt = optimizer_cls(maxfun=500 * epochs)
        history: List[float] = []

        def loss(params: np.ndarray) -> float:
            # Update last unitary
            self.unitaries[-1][0] = _random_qubit_unitary(self.arch[-1])  # placeholder
            # Compute predictions
            preds = [self.feedforward([(s, None)])[-1][-1] for s, _ in samples]
            # Compute MSE of overlaps
            mse = 0.0
            for pred, (_, target) in zip(preds, samples):
                mse += (1 - self.state_fidelity(pred, target))
            return mse / len(samples)

        for _ in range(epochs):
            params = opt.optimize(loss, 0)  # 0‑dimensional placeholder
            history.append(loss(params))
        return history


def random_training_data(target_unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate quantum training data using the target unitary."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    num_qubits = target_unitary.num_qubits
    for _ in range(samples):
        state = Statevector.from_label("0" * num_qubits)
        dataset.append((state, target_unitary @ state))
    return dataset


__all__ = [
    "GraphQNNHybridQML",
    "autoencoder_circuit",
    "random_training_data",
]
