"""Quantum self‑attention module using PennyLane.

The class builds a variational circuit that:
  * Encodes the classical attention vector into an n‑qubit state via
    an angle embedding.
  * Applies a stack of parameterized rotations and entangling gates
    determined by `rotation_params` and `entangle_params`.
  * Returns the measurement probabilities and the final statevector.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import numpy as np

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _create_device(n_qubits: int, shots: int = 1024):
    return qml.device("default.qubit", wires=n_qubits, shots=shots)

def _angle_embedding(vec: np.ndarray, wires: List[int]):
    """Map a real vector onto rotation angles of the qubits."""
    for w, theta in zip(wires, vec):
        qml.RY(theta, wires=w)

def _entangling_layer(params: np.ndarray, wires: List[int]):
    """Apply a layer of CNOTs with rotation parameters."""
    for w, theta in zip(wires[:-1], params):
        qml.RZ(theta, wires=w)
        qml.CNOT(wires=[w, wires[w+1]])

# --------------------------------------------------------------------------- #
# Quantum self‑attention circuit
# --------------------------------------------------------------------------- #
class HybridQuantumSelfAttention:
    """
    Variational circuit that implements a quantum‑enhanced self‑attention.
    """
    def __init__(
        self,
        n_qubits: int,
        qnn_arch: List[int],
        shots: int = 1024,
    ):
        self.n_qubits = n_qubits
        self.qnn_arch = qnn_arch
        self.device = _create_device(n_qubits, shots=shots)

        @qml.qnode(self.device, interface="numpy")
        def circuit(rotation_params: np.ndarray, entangle_params: np.ndarray):
            # 1. Angle embedding of the classical attention vector
            _angle_embedding(rotation_params[:self.n_qubits], wires=range(self.n_qubits))
            # 2. Entangling layer
            _entangling_layer(entangle_params, wires=range(self.n_qubits))
            # 3. Measurement of all qubits
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        *,
        shots: int | None = None,
    ) -> dict:
        """
        Execute the variational circuit and return measurement probabilities
        along with the final statevector.
        """
        if shots is not None:
            self.device.shots = shots

        probs = self.circuit(rotation_params, entangle_params)

        # Retrieve the final statevector for fidelity analysis
        statevector = self.device.execute(
            [self.circuit], [rotation_params, entangle_params], shots=0
        )[0]
        # Build a trivial fidelity graph (single node)
        graph = nx.Graph()
        graph.add_node(0)
        return {
            "probs": probs,
            "statevector": statevector,
            "fidelity_graph": graph,
        }

__all__ = ["HybridQuantumSelfAttention"]
