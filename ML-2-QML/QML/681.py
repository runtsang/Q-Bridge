import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

import pennylane as qml
import pennylane.numpy as np
import networkx as nx
import torch

Tensor = torch.Tensor

def _random_qubit_unitary(num_qubits: int) -> qml.QubitUnitary:
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    unitary = np.linalg.qr(matrix)[0]
    return qml.QubitUnitary(unitary, wires=range(num_qubits))

def _random_qubit_state(num_qubits: int) -> qml.Statevector:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return qml.Statevector(vec, wires=range(num_qubits))

def random_training_data(unitary: qml.QubitUnitary, samples: int) -> List[Tuple[qml.Statevector, qml.Statevector]]:
    dataset: List[Tuple[qml.Statevector, qml.Statevector]] = []
    for _ in range(samples):
        state = _random_qubit_state(len(unitary.wires))
        output = unitary @ state
        dataset.append((state, output))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Tuple[qml.QNode, np.ndarray]]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Tuple[qml.QNode, np.ndarray]] = []

        for _ in range(num_outputs):
            wires = list(range(num_inputs + 1))
            dev = qml.device("default.qubit", wires=wires)

            @qml.qnode(dev, interface="torch")
            def layer_circuit(x, *params):
                # Encode input state
                for i, w in enumerate(wires[:-1]):
                    qml.RY(x[i], wires=w)
                # Parametrised rotations
                for i, w in enumerate(wires):
                    qml.RX(params[i], wires=w)
                    qml.RZ(params[i + num_inputs + 1], wires=w)
                # Entanglement
                for i in range(num_inputs):
                    qml.CNOT(wires[i], wires[i + 1])
                return qml.state()

            num_params = (num_inputs + 1) * 2
            params = np.random.uniform(0, 2 * np.pi, size=num_params, requires_grad=True)
            layer_ops.append((layer_circuit, params))
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qml.Statevector, keep: Sequence[int]) -> qml.Statevector:
    return state.trace_out([i for i in range(len(state.wires)) if i not in keep])

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Tuple[qml.QNode, np.ndarray]]],
                   layer: int, input_state: qml.Statevector) -> qml.Statevector:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = input_state

    # Apply the first unitary
    circ, params = unitaries[layer][0]
    state = circ(state, *params)

    # If multiple outputs, apply remaining unitaries and combine
    if num_outputs > 1:
        for circ, params in unitaries[layer][1:]:
            state = circ(state, *params)

    # Keep only the output qubits
    keep = list(range(num_inputs, num_inputs + num_outputs))
    return _partial_trace_keep(state, keep)

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Tuple[qml.QNode, np.ndarray]]],
                samples: Iterable[Tuple[qml.Statevector, qml.Statevector]]):
    stored_states: List[List[qml.Statevector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qml.Statevector, b: qml.Statevector) -> float:
    overlap = np.vdot(a, b)
    return float(np.abs(overlap) ** 2)

def fidelity_adjacency(
    states: Sequence[qml.Statevector],
    threshold: float,
    *,
    secondary: Optional[float] = None,
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

class GraphQNN:
    """
    Quantum‑classical hybrid graph neural network.

    Mirrors the API of the classical GraphQNN while executing on a
    Pennylane backend.  The class stores a list of parameterised QNodes,
    one per output qubit of each layer.  Training is performed with
    PennyLane's autograd, and a hybrid loss combines a fidelity term
    between predicted and target states.
    """

    def __init__(
        self,
        arch: Sequence[int],
        device: str = "default.qubit",
        *,
        loss_weight: float = 1.0,
        early_stopping: Optional[bool] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.arch = list(arch)
        self.device = device
        self.loss_weight = loss_weight
        self.early_stopping = early_stopping
        self.checkpoint_path = checkpoint_path

        # Randomly initialise parameters for the quantum circuit
        self.unitaries: List[List[Tuple[qml.QNode, np.ndarray]]] = [[]]
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[Tuple[qml.QNode, np.ndarray]] = []

            for _ in range(num_outputs):
                wires = list(range(num_inputs + 1))
                dev = qml.device(self.device, wires=wires)

                @qml.qnode(dev, interface="torch")
                def layer_circuit(x, *params):
                    # Encode input state
                    for i, w in enumerate(wires[:-1]):
                        qml.RY(x[i], wires=w)
                    # Parametrised rotations
                    for i, w in enumerate(wires):
                        qml.RX(params[i], wires=w)
                        qml.RZ(params[i + num_inputs + 1], wires=w)
                    # Entanglement
                    for i in range(num_inputs):
                        qml.CNOT(wires[i], wires[i + 1])
                    return qml.state()

                num_params = (num_inputs + 1) * 2
                params = np.random.uniform(0, 2 * np.pi, size=num_params, requires_grad=True)
                layer_ops.append((layer_circuit, params))
            self.unitaries.append(layer_ops)

        # Graph for fidelity‑based adjacency (used in analysis)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(len(self.arch)))

    def feedforward(self, samples: Iterable[Tuple[qml.Statevector, qml.Statevector]]):
        """Return all intermediate states for the provided samples."""
        return feedforward(self.arch, self.unitaries, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[qml.Statevector],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def hybrid_loss(
        self,
        output_states: List[qml.Statevector],
        target_states: List[qml.Statevector],
    ) -> Tensor:
        """
        Compute a weighted sum of a fidelity loss between the predicted
        and target quantum states.
        """
        fid_sum = sum(state_fidelity(out, tgt) for out, tgt in zip(output_states, target_states))
        fidelity_loss = 1.0 - fid_sum / len(output_states)
        return torch.tensor(fidelity_loss, dtype=torch.float32, device="cpu")

    def train(
        self,
        dataset: Iterable[Tuple[qml.Statevector, qml.Statevector]],
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 8,
        verbose: bool = True,
    ) -> List[float]:
        """Placeholder training loop.  Actual optimisation is omitted."""
        loss_history: List[float] = []
        data = list(dataset)

        for epoch in range(epochs):
            epoch_loss = 0.0

            np.random.shuffle(data)

            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                output_states = [self._forward(sample) for sample, _ in batch]
                target_states = [tgt for _, tgt in batch]

                loss = self.hybrid_loss(output_states, target_states)
                loss.backward()  # gradients are computed but not applied

                epoch_loss += loss.item() * len(batch)

            epoch_loss /= len(data)
            loss_history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")

            if self.early_stopping and epoch > 0 and loss_history[-1] > loss_history[-2]:
                if verbose:
                    print("Early stopping triggered.")
                break

        return loss_history

    def _forward(self, sample: qml.Statevector) -> qml.Statevector:
        """Apply the full network to a single input state."""
        state = sample
        for layer in range(1, len(self.arch)):
            state = _layer_channel(self.arch, self.unitaries, layer, state)
        return state

    def export_surrogate(self) -> List[Tensor]:
        """Return a list of classical weights derived from the first layer."""
        surrogate = []
        for circ, params in self.unitaries[1]:
            w = torch.tensor(params[: self.arch[0] + 1], dtype=torch.float32)
            surrogate.append(w)
        return surrogate

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: qml.QubitUnitary, samples: int):
        return random_training_data(unitary, samples)

    @staticmethod
    def state_fidelity(a: qml.Statevector, b: qml.Statevector) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qml.Statevector],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
