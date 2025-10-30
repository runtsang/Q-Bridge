"""Quantum graph neural network utilities mirroring the classical API.

This module implements the same public functions and classes as the
classical GraphQNN, but with quantum backends.  It is intentionally
compatible with the hybrid constructor used in the classical module.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import torchquantum as tq

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Quantum utilities – random unitary generation and basic operations
# --------------------------------------------------------------------------- #

def _random_unitary(dim: int) -> torch.Tensor:
    """Generate a random unitary matrix via QR decomposition."""
    mat = torch.randn(dim, dim, dtype=torch.complex64) + 1j * torch.randn(dim, dim, dtype=torch.complex64)
    q, _ = torch.linalg.qr(mat)
    return q

def _tensor_zero(num_qubits: int) -> torch.Tensor:
    return torch.zeros(2 ** num_qubits, dtype=torch.complex64)

# --------------------------------------------------------------------------- #
#  Dataset generation – identical to the classical version but returns complex
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Same as the classical version but returns complex amplitudes."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

def random_training_data(unitary: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate input–output pairs for a target unitary."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        state = torch.randn(unitary.size(0), dtype=torch.complex64)
        state /= torch.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Return a random quantum architecture and training data."""
    unitaries: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        dim = 2 ** out_f
        unitaries.append(_random_unitary(dim))
    target_unitary = unitaries[-1]
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), unitaries, training_data, target_unitary

# --------------------------------------------------------------------------- #
#  Feed‑forward – apply a sequence of unitaries to an input state
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Return the state trajectory for each sample."""
    stored: List[List[torch.Tensor]] = []
    for state, _ in samples:
        trajectory = [state]
        current = state
        for unitary in unitaries:
            current = unitary @ current
            trajectory.append(current)
        stored.append(trajectory)
    return stored

# --------------------------------------------------------------------------- #
#  Fidelity helpers – identical to the classical implementation
# --------------------------------------------------------------------------- #

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared absolute overlap of two normalized quantum states."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.abs(torch.dot(a_norm.conj(), b_norm)).item() ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities."""
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
#  Quantum self‑attention – Qiskit implementation
# --------------------------------------------------------------------------- #

class SelfAttentionQuantum:
    """Quantum self‑attention block implemented with Qiskit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> Dict[str, int]:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

# --------------------------------------------------------------------------- #
#  Quantum fully‑connected layer – Qiskit implementation
# --------------------------------------------------------------------------- #

class FCLQuantum:
    """Parameterized quantum circuit that emulates a fully‑connected layer."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        self.theta = theta
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

# --------------------------------------------------------------------------- #
#  Quantum regression dataset & model – TorchQuantum implementation
# --------------------------------------------------------------------------- #

class RegressionDataset(tq.QuantumDataset):
    """Quantum dataset mirroring the classical regression example."""
    def __init__(self, samples: int, num_wires: int):
        super().__init__()
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": torch.tensor(self.states[index], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[index], dtype=torch.float32)}

class QModel(tq.QuantumModule):
    """Quantum regression network that mirrors the classical model."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
#  Hybrid network builder – assemble classical & quantum layers
# --------------------------------------------------------------------------- #

class HybridGraphQNN:
    """Hybrid quantum‑classical network built from a sequence of layer specs.

    The layer spec is identical to the classical counterpart.  When a
    quantum layer is encountered the state is converted to a numpy array,
    passed to the quantum layer, and the resulting vector is converted
    back to a torch tensor.
    """
    def __init__(self, layer_specs: Sequence[Dict[str, Any]]):
        self.layers = []
        for spec in layer_specs:
            if spec["type"] == "classical":
                self.layers.append(nn.Linear(*spec["shape"]))
            elif spec["type"] == "quantum":
                self.layers.append(spec["layer"])
            else:
                raise ValueError(f"Unknown layer type {spec['type']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Module):
                out = layer(out)
            else:
                out_np = layer.run(out.numpy())
                out = torch.tensor(out_np, dtype=out.dtype, device=out.device)
        return out

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "SelfAttentionQuantum",
    "FCLQuantum",
    "RegressionDataset",
    "QModel",
    "HybridGraphQNN",
]
