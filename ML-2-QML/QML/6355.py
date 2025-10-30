"""
Quantum circuit utilities for the hybrid QuantumNATEnhanced model.

The circuit is implemented with PennyLane and can be executed on a
CPU simulator or a real quantum device (e.g. IBM Q, IonQ, Braket).
"""

import pennylane as qml
import torch
import numpy as np
from typing import Tuple

def variational_circuit(x: torch.Tensor,
                        params: torch.Tensor,
                        backend: str = 'default.qubit',
                        device: str = 'cpu',
                        n_qubits: int = 4,
                        shots: int = 1000) -> torch.Tensor:
    """
    Executes a simple variational quantum circuit on each sample in `x`.

    Args:
        x: Tensor of shape (batch, features). The first `n_qubits`
           features are used as input angles for Ry rotations.
        params: Tensor of shape (n_layers, n_qubits, 3) containing
                rotation angles for the variational layers.
        backend: PennyLane device name (e.g. 'default.qubit',
                 'qiskit.ibmq_qasm_simulator', 'braket.aws.braket').
        device: 'cpu' or 'qpu'. When 'cpu' a classical simulation
                is performed. When 'qpu' the specified backend is used.
        n_qubits: Number of qubits in the circuit.
        shots: Number of shots for expectation estimation on a real device.

    Returns:
        Tensor of shape (batch, n_qubits) containing expectation
        values of PauliZ for each qubit.
    """
    # Choose device
    dev = qml.device(backend,
                     wires=n_qubits,
                     shots=None if device == 'cpu' else shots)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(sample, param):
        # Initialize qubits with Ry rotations from the sample
        for q in range(n_qubits):
            qml.RY(sample[q], wires=q)

        # Variational layers
        for layer in range(param.shape[0]):
            for q in range(n_qubits):
                qml.RX(param[layer, q, 0], wires=q)
                qml.RY(param[layer, q, 1], wires=q)
                qml.RZ(param[layer, q, 2], wires=q)
            # Entangling layer (cyclic CNOTs)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        # Measurement of PauliZ expectation values
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    batch = x.shape[0]
    outputs = []

    for i in range(batch):
        # Ensure the sample has the correct number of qubits
        sample = x[i, :n_qubits]
        out = circuit(sample, params)
        outputs.append(out)

    return torch.stack(outputs)

__all__ = ["variational_circuit"]
