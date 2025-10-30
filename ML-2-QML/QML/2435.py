"""Quantum‑only version of the FraudGraphHybrid model.

This QML module implements a quantum circuit that mirrors the photonic
layer structure used in the classical model.  The circuit is built
with a variational gate set that is *not* tied to the photonic
hardware, but we preserve the user‑defined parameter format
(FraudLayerParameters).  The circuit can be executed on any
simulator or hardware backend that supports a “continuous‑variable”
style or a standard qubit‑based variational circuit.

The design keeps the quantum and classical parts separate so that
the quantum circuit can be used independently or as a twin of the
classical network.  The output of the circuit is a statevector
which can be fed into a fidelity‑based graph or used for
measurement‑based loss functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pennylane as qml
import numpy as np
import networkx as nx
from qiskit.quantum_info import Statevector


# --------------------------------------------------------------------------- #
# Parameter dataclass (identical to classical side)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, reused for the QML circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


# --------------------------------------------------------------------------- #
# Helper functions for clipping and gate mapping
# --------------------------------------------------------------------------- #

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, clip: bool) -> None:
    """Map the photonic layer parameters to a qubit‑based variational circuit."""
    # Entangling gate analogous to BSgate
    qml.CNOT(wires=[wires[0], wires[1]])

    # Phase rotations (Rgate)
    for i, phase in enumerate(params.phases):
        qml.RZ(_clip(phase, 5.0), wires=wires[i])

    # Squeezing gate analogue (Sgate) -> Rot
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Rot(_clip(r, 5.0), 0.0, _clip(phi, 5.0), wires=wires[i])

    # Second entangling gate
    qml.CNOT(wires=[wires[0], wires[1]])

    # Phase rotations again
    for i, phase in enumerate(params.phases):
        qml.RZ(_clip(phase, 5.0), wires=wires[i])

    # Displacement gate analogue (Dgate) -> RY
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RY(_clip(r, 5.0), wires=wires[i])

    # Kerr gate analogue (Kgate) -> RZ
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1.0), wires=wires[i])


# --------------------------------------------------------------------------- #
# Circuit construction
# --------------------------------------------------------------------------- #

def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    device: qml.Device,
) -> qml.QNode:
    """Return a QNode that implements the photonic‑style fraud‑detection circuit."""
    @qml.qnode(device)
    def circuit():
        # Initialise to |00>
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)

        # Apply input layer without clipping
        _apply_layer([0, 1], input_params, clip=False)

        # Apply subsequent layers with clipping
        for params in layers:
            _apply_layer([0, 1], params, clip=True)

        # Return the full statevector for downstream use
        return qml.state()

    return circuit


# --------------------------------------------------------------------------- #
# Random parameter generation for testing
# --------------------------------------------------------------------------- #

def random_layer_params() -> FraudLayerParameters:
    """Generate a random set of parameters for one layer."""
    return FraudLayerParameters(
        bs_theta=np.random.uniform(-np.pi, np.pi),
        bs_phi=np.random.uniform(-np.pi, np.pi),
        phases=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
        squeeze_r=(np.random.uniform(0, 2), np.random.uniform(0, 2)),
        squeeze_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
        displacement_r=(np.random.uniform(0, 2), np.random.uniform(0, 2)),
        displacement_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
        kerr=(np.random.uniform(-1, 1), np.random.uniform(-1, 1)),
    )


def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random QNN architecture and corresponding training data."""
    # Build random parameters for each layer
    layers = [random_layer_params() for _ in range(len(qnn_arch) - 1)]

    # Create a device that can return statevectors
    dev = qml.device("default.qubit", wires=2)

    # Build the circuit
    circuit = build_fraud_detection_circuit(layers[0], layers[1:], dev)

    # Generate training data by sampling random input states
    training_data = []
    for _ in range(samples):
        # Random input state via random unitary on |00>
        unitary = qml.math.random_unitary(2)
        qml.apply(unitary, wires=[0, 1])
        input_state = qml.state()
        target_state = circuit()
        training_data.append((input_state, target_state))

    return qnn_arch, layers, training_data, circuit


# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return the absolute squared overlap between two pure statevectors."""
    return abs(np.vdot(a.data, b.data)) ** 2


def fidelity_adjacency(states: Sequence[Statevector], threshold: float,
                      *, secondary: float | None = None,
                      secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
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
# Hybrid utilities
# --------------------------------------------------------------------------- #

def hybrid_fidelity_graph(
    qnn_arch: Sequence[int],
    layers: Sequence[FraudLayerParameters],
    training_data: Sequence[tuple[Statevector, Statevector]],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Return a fidelity‑based graph for the input states of a QNN."""
    input_states = [Statevector(data) for data, _ in training_data]
    return fidelity_adjacency(input_states, threshold,
                              secondary=secondary,
                              secondary_weight=secondary_weight)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "random_layer_params",
    "random_network",
    "state_fidelity",
    "fidelity_adjacency",
    "hybrid_fidelity_graph",
]
