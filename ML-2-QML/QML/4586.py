"""Hybrid quantum model that mirrors the classical encoder and introduces a quantum variational ansatz.

The model can be used for classification or regression and is designed to be dropped into
the same experiment pipeline as the classical counterpart.  The quantum layer uses a
data‑upload style GeneralEncoder, a random layer, and parameterised RX/RY rotations
followed by a MeasureAll readout.  The head is a classical linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from typing import Iterable, Tuple


# --------------------------------------------------------------------------- #
#  Data‑upload style encoder (same as classical but in quantum device)
# --------------------------------------------------------------------------- #
class QuantumEncoder(tq.QuantumModule):
    """
    Encoder that maps classical feature vectors to a quantum state
    using the GeneralEncoder with an Ry‑style pulse schedule.
    """
    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )

    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> None:
        self.encoder(qdev, data)


# --------------------------------------------------------------------------- #
#  Variational layer (RandomLayer + trainable rotations)
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """
    Applies a RandomLayer followed by a trainable RX+RY on each wire.
    """
    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)


# --------------------------------------------------------------------------- #
#  Hybrid quantum model definition
# --------------------------------------------------------------------------- #
class HybridQuantumModel(tq.QuantumModule):
    """
    Quantum‑style hybrid classifier/regressor.

    Parameters
    ----------
    num_wires : int
        Number of qubits / wires in the device.
    output_dim : int, default 2
        Size of the linear head (2 for classification, 1 for regression).
    regression : bool, default False
        If True, the model outputs a scalar regression value.
    """

    def __init__(self, num_wires: int, output_dim: int = 2, regression: bool = False) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = QuantumEncoder(num_wires)
        self.q_layer = QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, output_dim)
        self.regression = regression

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Input features of shape (batch, num_wires).

        Returns
        -------
        torch.Tensor
            Logits (classification) or regression value.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data into qubits
        self.encoder(qdev, state_batch)
        # Apply variational layer
        self.q_layer(qdev)
        # Measure all qubits
        features = self.measure(qdev)
        # Linear head
        return self.head(features).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Utility: build a quantum classifier circuit (metadata only)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[tq.QuantumCircuit, Iterable, Iterable, list[tq.SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational parameters,
    mirroring the classical metadata.

    Returns
    -------
    Tuple[tq.QuantumCircuit, Iterable, Iterable, list[tq.SparsePauliOp]]
        (circuit, encoding params, weight params, observables)
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables
