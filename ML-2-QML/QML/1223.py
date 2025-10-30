"""Quantum utilities for hybrid graph neural networks.

This module implements a variational quantum circuit that receives a classical
latent vector as control data.  The circuit is built with PennyLane and
returns a pure state vector that can be compared to a target unitary via
fidelity.  The module also provides helpers for generating random training
data and for constructing a fidelity‑based adjacency graph.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml
import torch
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Variational circuit
# --------------------------------------------------------------------------- #
def _create_device(num_qubits: int) -> qml.Device:
    return qml.device("default.qubit", wires=num_qubits)


def quantum_variational(latent: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Encode the classical latent vector into a quantum state and apply a
    simple variational ansatz.

    Parameters
    ----------
    latent : torch.Tensor
        1‑D tensor of length ``num_qubits`` containing classical data.
    params : torch.Tensor
        1‑D tensor of variational angles.

    Returns
    -------
    torch.Tensor
        Statevector as a 1‑D torch tensor of complex dtype.
    """
    num_qubits = latent.shape[0]
    dev = _create_device(num_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(latent_vec: torch.Tensor, var_params: torch.Tensor) -> torch.Tensor:
        # Encode classical data via RY rotations
        for i in range(num_qubits):
            qml.RY(latent_vec[i], wires=i)
        # Variational layer (single layer for brevity)
        for i, angle in enumerate(var_params):
            qml.RZ(angle, wires=i % num_qubits)
            qml.RX(angle, wires=(i + 1) % num_qubits)
        return qml.state()

    state = circuit(latent, params)
    return state  # complex torch tensor


# --------------------------------------------------------------------------- #
# 2. Fidelity loss
# --------------------------------------------------------------------------- #
def fidelity_loss(state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute overlap (fidelity) between two pure states.

    Parameters
    ----------
    state_a, state_b : torch.Tensor
        1‑D complex tensors representing statevectors.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the fidelity.
    """
    overlap = torch.vdot(state_a, state_b)
    return torch.abs(overlap) ** 2


# --------------------------------------------------------------------------- #
# 3. Random unitary and training data
# --------------------------------------------------------------------------- #
def random_quantum_unitary(num_qubits: int) -> torch.Tensor:
    """
    Generate a random unitary and apply it to the all‑zero state.

    Returns
    -------
    torch.Tensor
        Statevector of the target pure state.
    """
    # Generate random unitary
    U_np = qml.math.random_unitary(2 ** num_qubits)
    U = torch.from_numpy(U_np).to(torch.complex128)
    # All‑zero state
    zero_state = torch.zeros(2 ** num_qubits, dtype=torch.complex128)
    zero_state[0] = 1.0
    target = U @ zero_state
    return target


def random_quantum_training_data(target_state: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate random latent vectors and pair them with the same target state.

    Parameters
    ----------
    target_state : torch.Tensor
        Target pure statevector.
    samples : int
        Number of training examples.

    Returns
    -------
    list[tuple[torch.Tensor, torch.Tensor]]
        List of (latent, target_state) pairs.
    """
    num_qubits = target_state.shape[0].bit_length() - 1
    data = []
    for _ in range(samples):
        latent = torch.randn(num_qubits, dtype=torch.float32)
        data.append((latent, target_state))
    return data


# --------------------------------------------------------------------------- #
# 4. Fidelity‑based adjacency graph
# --------------------------------------------------------------------------- #
def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = fidelity_loss(state_i, state_j).item()
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = [
    "quantum_variational",
    "fidelity_loss",
    "random_quantum_unitary",
    "random_quantum_training_data",
    "fidelity_adjacency",
]
