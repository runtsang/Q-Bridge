from __future__ import annotations

import numpy as np
import torch
import networkx as nx
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

# --------------------------------------------------------------------
# Data generation – quantum version of the superposition dataset.
# --------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Returns raw state vectors and target labels.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset that emits quantum state vectors and scalar targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------
# Helper: fidelity between two statevectors (numpy).
# --------------------------------------------------------------------
def fidelity_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Absolute squared overlap between two pure state vectors.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.abs(np.vdot(a_norm, b_norm)) ** 2)

def fidelity_adjacency(states: list[np.ndarray], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """
    Build a weighted adjacency graph from state fidelities.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            fid = fidelity_np(states[i], states[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------
# Quantum estimator – mirrors EstimatorQNN.py using Qiskit primitives.
# --------------------------------------------------------------------
def EstimatorQNN(num_features: int = 2, num_wires: int = 2) -> QiskitEstimatorQNN:
    """
    Construct a Qiskit EstimatorQNN that encodes the input features
    via Ry rotations, applies a random layer, and measures Pauli‑Z
    expectation on all qubits.  The circuit is parameterized
    with input and weight parameters that can be trained by the
    StatevectorEstimator primitive.
    """
    # Input parameters
    inputs = ParameterVector("input", num_features)
    # Weight parameters – one per qubit for a simple variational layer
    weights = ParameterVector("weight", num_wires)

    qc = QuantumCircuit(num_wires)
    # Encode inputs
    for i, param in enumerate(inputs):
        qc.ry(param, i)
    # Entangling layer
    for i in range(num_wires - 1):
        qc.cx(i, i + 1)
    # Variational layer
    for i, param in enumerate(weights):
        qc.rx(param, i)
        qc.rz(param * 0.5, i)  # extra rotation for richer expressivity

    # Observable – sum of PauliZ on all qubits
    observable = SparsePauliOp.from_list([("Z" * num_wires, 1)])

    # Primitive
    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=inputs,
        weight_params=weights,
        estimator=estimator,
    )
    return estimator_qnn

# --------------------------------------------------------------------
# Quantum sampler – mirrors SamplerQNN.py using Qiskit primitives.
# --------------------------------------------------------------------
def SamplerQNN(num_qubits: int = 2) -> QiskitSamplerQNN:
    """
    Construct a Qiskit SamplerQNN that implements a simple 2‑qubit
    parameterized circuit and returns a probability distribution
    over the computational basis.
    """
    inputs = ParameterVector("input", num_qubits)
    weights = ParameterVector("weight", 4)

    qc = QuantumCircuit(num_qubits)
    # Input rotations
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    # Entanglement
    qc.cx(0, 1)
    # Variational rotations
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = StatevectorSampler()
    sampler_qnn = QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )
    return sampler_qnn

__all__ = [
    "EstimatorQNN",
    "SamplerQNN",
    "RegressionDataset",
    "generate_superposition_data",
    "fidelity_adjacency",
]
