"""Quantum sampler network with a multi‑layer entangling ansatz.

The original seed implemented a 2‑qubit circuit with a fixed depth.  This module expands the design to support an arbitrary number of qubits and entangling layers, providing a richer variational space while keeping the public API identical to the classical counterpart.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pennylane as qml
from pennylane import numpy as np
from pennylane.qiskit import QiskitDevice
from pennylane import qnet
from pennylane import qml
from pennylane import qml
from pennylane import qml
from pennylane import qml

# Import the Qiskit SamplerQNN for compatibility
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

def SamplerQNN(
    n_qubits: int = 3,
    depth: int = 2,
    entanglement: str = "full",
) -> QiskitSamplerQNN:
    """
    Construct a parameterised quantum sampler with *n_qubits* and *depth* entangling layers.

    Parameters
    ----------
    n_qubits : int, default 3
        Number of qubits in the circuit.
    depth : int, default 2
        Number of alternating rotation‑entanglement blocks.
    entanglement : str, {"full", "circular", "linear"}, default "full"
        Entanglement pattern for the controlled‑NOT gates.

    Returns
    -------
    QiskitSamplerQNN
        A ready‑to‑use Qiskit SamplerQNN instance.

    Notes
    -----
    The circuit is built using Pennylane to automatically generate the required
    ParameterVector objects.  These vectors are then passed to the Qiskit
    SamplerQNN wrapper for integration with the Qiskit Machine Learning
    framework.
    """

    # Create parameter vectors for inputs and weights
    input_params = qml.measurements.ParameterVector("input", n_qubits)
    weight_params = qml.measurements.ParameterVector("weight", n_qubits * depth * 2)

    # Build the variational circuit
    qc = qml.QNode(
        lambda *params: _build_circuit(params, n_qubits, depth, entanglement),
        device=qml.device("qiskit", backend="statevector_simulator"),
    )

    # Convert the QNode to a Qiskit QuantumCircuit
    circuit = qc.to_qiskit()

    # Instantiate the Qiskit SamplerQNN
    sampler = Sampler()
    sampler_qnn = QiskitSamplerQNN(
        circuit=circuit,
        input_params=input_params,
        weight_params=weight_params,
        sampler=sampler,
    )
    return sampler_qnn


def _build_circuit(
    params: Iterable[float],
    n_qubits: int,
    depth: int,
    entanglement: str,
) -> None:
    """Internal helper that constructs the circuit from a flattened parameter list."""
    params = np.array(params)

    # Split parameters into rotations and entanglement layers
    rot_params = params[: n_qubits * depth * 2]
    weight_idx = 0

    for d in range(depth):
        # Rotation layer
        for q in range(n_qubits):
            qml.RY(rot_params[weight_idx], wires=q)
            weight_idx += 1
            qml.RZ(rot_params[weight_idx], wires=q)
            weight_idx += 1

        # Entanglement layer
        if entanglement == "full":
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        elif entanglement == "circular":
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        elif entanglement == "linear":
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        else:
            raise ValueError(f"Unknown entanglement pattern: {entanglement}")

__all__ = ["SamplerQNN"]
