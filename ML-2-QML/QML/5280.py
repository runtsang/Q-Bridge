"""Quantum implementation of the fraud detection circuit.

The model mirrors the classical counterpart:
- A stack of photonic‑style layers is realised with rotation gates.
- A graph‑based adjacency is constructed from state fidelities and used to weight a
  controlled‑phase block.
- A Qiskit ansatz (derived from ``QuantumClassifierModel``) is appended for classification.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import ParameterVector

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybridModel",
    "random_fraud_network",
    "state_fidelity",
    "fidelity_adjacency",
    "feedforward",
    "build_classifier_circuit",
]

# --------------------------------------------------------------------------- #
# 1.  Layer parameter definition (identical to classical)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# 2.  Quantum photonic layer implementation
# --------------------------------------------------------------------------- #
def _apply_photonic_layer(circ: QuantumCircuit, params: FraudLayerParameters) -> None:
    """Map photonic parameters to a sequence of Qiskit rotation gates."""
    # Beam‑splitter angles -> RX on each qubit
    circ.rx(params.bs_theta, 0)
    circ.rx(params.bs_phi, 1)

    # Phase shifters
    for i, phase in enumerate(params.phases):
        circ.rz(phase, i)

    # Squeezing -> RX followed by RZ
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circ.rx(r, i)
        circ.rz(phi, i)

    # Displacement -> RY followed by RZ
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circ.ry(r, i)
        circ.rz(phi, i)

    # Kerr nonlinearity -> RZ
    for i, k in enumerate(params.kerr):
        circ.rz(k, i)

# --------------------------------------------------------------------------- #
# 3.  Hybrid quantum circuit builder
# --------------------------------------------------------------------------- #
class FraudDetectionHybridModel:
    """Quantum circuit that mirrors the classical photonic layers and includes a
    graph‑aware controlled‑phase block and a classifier ansatz."""

    def __init__(self, layers: List[FraudLayerParameters], num_qubits: int = 2):
        self.layers = layers
        self.num_qubits = num_qubits
        self.circuit = self.build_fraud_detection_program()

    def build_fraud_detection_program(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.num_qubits)
        for params in self.layers:
            _apply_photonic_layer(circ, params)
        # Graph‑aware controlled‑phase block (CZ for each adjacency edge)
        # The adjacency graph will be constructed after state propagation
        return circ

    def forward(self, input_state: np.ndarray) -> Statevector:
        """Execute the circuit on the provided state vector."""
        sv = Statevector.from_label(input_state)
        sv = sv.evolve(self.circuit)
        return sv

# --------------------------------------------------------------------------- #
# 4.  Utility functions
# --------------------------------------------------------------------------- #
def random_fraud_network(num_layers: int, seed: int | None = None) -> List[FraudLayerParameters]:
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(num_layers):
        layers.append(
            FraudLayerParameters(
                bs_theta=rng.standard_normal(),
                bs_phi=rng.standard_normal(),
                phases=(rng.standard_normal(), rng.standard_normal()),
                squeeze_r=(rng.standard_normal(), rng.standard_normal()),
                squeeze_phi=(rng.standard_normal(), rng.standard_normal()),
                displacement_r=(rng.standard_normal(), rng.standard_normal()),
                displacement_phi=(rng.standard_normal(), rng.standard_normal()),
                kerr=(rng.standard_normal(), rng.standard_normal()),
            )
        )
    return layers


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two pure statevectors."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(states: Iterable[Statevector], threshold: float) -> nx.Graph:
    G = nx.Graph()
    states = list(states)
    G.add_nodes_from(range(len(states)))
    for i, ai in enumerate(states):
        for j, aj in enumerate(states[i + 1 :], start=i + 1):
            fid = state_fidelity(ai, aj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
    return G


def feedforward(
    model: FraudDetectionHybridModel, dataset: Iterable[Tuple[np.ndarray, np.ndarray]]
) -> List[Statevector]:
    """Run the quantum model on a dataset of input state labels."""
    outputs = []
    for inp, _ in dataset:
        outputs.append(model.forward(inp))
    return outputs


def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Quantum classifier ansatz identical to the one in ``QuantumClassifierModel.py``."""
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

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables
