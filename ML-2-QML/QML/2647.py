"""Hybrid quantum‑classical regression with graph‑based fidelity filtering and random‑unitary training data."""
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import networkx as nx
import qutip as qt
from scipy.spatial.distance import pdist, squareform

# --------------------------------------------------------------------------- #
# 1. Data generation using random unitaries
# --------------------------------------------------------------------------- #
def generate_random_unitary_data(
    num_qubits: int,
    samples: int,
    fidelity_threshold: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quantum states via random unitary evolution of a
    single‑qubit state and produce the label as the expectation
    of a Pauli‑Z operator on the first qubit.
    """
    # Initial state |0...0>
    init_state = qt.basis(2 ** num_qubits, 0)

    # Random unitary
    random_unitary = qt.random_unitary(num_qubits)

    # Prepare states and labels
    states = []
    labels = []
    for _ in range(samples):
        state = random_unitary * init_state
        states.append(state.full().reshape(-1))
        # Expectation of Z on first qubit
        z_op = qt.tensor(qt.sigmaz(), *[qt.qeye(2)] * (num_qubits - 1))
        labels.append(qt.expect(z_op, state).real)

    states = np.array(states, dtype=np.complex64)
    labels = np.array(labels, dtype=np.float32)

    # Fidelity filtering
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fid = state_fidelity(qt.Qobj(states[i].reshape(-1, 1)), qt.Qobj(states[j].reshape(-1, 1)))
            if fid >= fidelity_threshold:
                G.add_edge(i, j)

    keep = [n for n in G.nodes if G.degree(n) < 2]
    return states[keep], labels[keep]


class RandomUnitaryDataset(torch.utils.data.Dataset):
    """
    Dataset that returns random‑unitary‑generated states and their labels.
    """
    def __init__(self, samples: int, num_qubits: int):
        self.states, self.labels = generate_random_unitary_data(num_qubits, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 2. Quantum encoder + hybrid model
# --------------------------------------------------------------------------- #
class QuantumEncoder(tq.QuantumModule):
    """
    Variational circuit that uses a random layer followed by trainable
    RX/RY gates.  The encoder is built on top of TorchQuantum
    (tq) and is fully differentiable.
    """
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)


class UnifiedQuantumRegression(tq.QuantumModule):
    """
    Hybrid model that runs a quantum circuit to produce feature vectors
    and then learns a residual with a classical network.
    """
    def __init__(self, num_features: int, num_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = QuantumEncoder(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.residual = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        residual = self.residual(state_batch)
        return self.head(features).squeeze(-1) + residual


__all__ = [
    "UnifiedQuantumRegression",
    "RandomUnitaryDataset",
    "generate_random_unitary_data",
]
