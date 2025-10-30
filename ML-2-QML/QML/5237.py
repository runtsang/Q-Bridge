from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
import networkx as nx
import itertools

# --------------------------------------------------------------------------- #
# 1. Dataset generation (inspired by QuantumRegression.py)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_qubits: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    Labels are sin(2 theta) * cos(phi).
    """
    omega_0 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_qubits), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_qubits: int):
        self.states, self.labels = generate_superposition_data(num_qubits, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# 2. Quantum classifier circuit (inspired by QuantumClassifierModel.py)
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# 3. Fidelity utilities (borrowed from GraphQNN but for statevectors)
# --------------------------------------------------------------------------- #

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Absolute squared overlap between two pure statevectors."""
    return abs((a.data.conj().T @ b.data).item()) ** 2

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
# 4. Unified FraudDetectionHybrid wrapper
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid:
    """
    Quantum implementation of the hybrid fraud‑detection pipeline.
    Encodes input amplitudes into a Qiskit circuit, runs a variational ansatz,
    measures Pauli‑Z on each qubit, and maps the expectation values through
    a classical linear head.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input dimensionality.
    depth : int
        Depth of the variational layers.
    threshold : float
        Fidelity threshold for graph construction.
    """

    def __init__(self, num_qubits: int, depth: int, threshold: float):
        self.num_qubits = num_qubits
        self.depth = depth
        self.threshold = threshold

        self.circuit, self.encoding_params, self.weights_params, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.head = nn.Linear(num_qubits, 1)

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of complex state vectors, execute the circuit on a
        statevector simulator, and return the linear head output.
        """
        bsz = state_batch.shape[0]
        simulator = Aer.get_backend("statevector_simulator")

        # Build a parameterized circuit for each batch element
        outputs = []
        for i in range(bsz):
            # Bind encoding parameters
            param_bindings = {str(p): state_batch[i, j].real for j, p in enumerate(self.encoding_params)}
            # The imaginary part is encoded as a phase on the initial state
            # (here we simply ignore it for brevity; a full implementation would
            # use a custom state‑prep routine).
            bound_circuit = self.circuit.bind_parameters(param_bindings)
            job = execute(bound_circuit, simulator)
            statevec = job.result().get_statevector(bound_circuit)
            sv = Statevector(statevec)
            # Measure expectation values of Pauli‑Z on each qubit
            exp_vals = np.array([sv.expectation_value(obs).real for obs in self.observables])
            outputs.append(exp_vals)

        exp_tensor = torch.tensor(outputs, dtype=torch.float32, device=state_batch.device)
        return self.head(exp_tensor).squeeze(-1)

    # --------------------------------------------------------------------- #
    # Graph utilities
    # --------------------------------------------------------------------- #

    def build_adjacency(self, state_vectors: Sequence[Statevector]) -> nx.Graph:
        """
        Construct a weighted graph from quantum state fidelities.
        """
        return fidelity_adjacency(
            states=state_vectors,
            threshold=self.threshold,
            secondary=self.threshold * 0.5,
            secondary_weight=0.3,
        )

    # --------------------------------------------------------------------- #
    # Convenience helpers
    # --------------------------------------------------------------------- #

    def random_network(self, arch: Sequence[int], samples: int):
        """
        Generate a synthetic quantum network and training data.
        (Placeholder – would normally construct random unitaries per layer.)
        """
        # For demonstration, return a list of random unitary matrices
        unitaries = [np.linalg.qr(np.random.randn(2 ** a, 2 ** a))[0] for a in arch]
        dataset = [(Statevector(np.random.rand(2 ** arch[0])), Statevector(np.random.rand(2 ** arch[-1]))) for _ in range(samples)]
        target_unitary = unitaries[-1]
        return arch, unitaries, dataset, target_unitary

__all__ = ["FraudDetectionHybrid"]
