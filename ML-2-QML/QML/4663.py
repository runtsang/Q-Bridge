"""Hybrid quantum‑classical classifier factory.

Defines a `HybridClassifierModel` that inherits from
`tq.QuantumModule`.  It builds a quantum ansatz identical to the
original Qiskit circuit and a linear head that maps the Z‑basis
expectation values to logits.  The API mirrors the classical
implementation so that a single `build_classifier_circuit` function can
be used in either branch.

A minimal dataset generator based on the superposition construction
is also included.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def generate_superposition_data(
    num_wires: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate quantum‑state samples for regression/ classification.

    The states are of the form
    ``cos(theta)|0…0> + exp(i phi) sin(theta)|1…1>``, with labels
    defined as ``sin(2 theta) cos(phi)`` for regression or a binary
    sign for classification.
    """
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)


class HybridClassifierModel(tq.QuantumModule):
    """Quantum implementation of the hybrid classifier."""

    def __init__(self, num_features: int, depth: int):
        super().__init__()
        self.num_features = num_features
        self.depth = depth

        # Build a Qiskit circuit that will be used for encoding and measurement
        encoding = ParameterVector("x", num_features)
        weights = ParameterVector("theta", num_features * depth)

        circuit = QuantumCircuit(num_features)
        for param, qubit in zip(encoding, range(num_features)):
            circuit.rx(param, qubit)

        index = 0
        for _ in range(depth):
            for qubit in range(num_features):
                circuit.ry(weights[index], qubit)
                index += 1
            for qubit in range(num_features - 1):
                circuit.cz(qubit, qubit + 1)

        self.circuit = circuit
        self.encoding = list(encoding)
        self.weights = list(weights)
        self.observables = [
            SparsePauliOp(f"I" * i + "Z" + "I" * (num_features - i - 1))
            for i in range(num_features)
        ]

        self.weight_sizes = [len(weights)]

        # Linear head to map the measurement vector to logits
        self.head = nn.Linear(num_features, 2)

        # Encoder for classical data – uses a general RX encoding
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(num_features)
            ]
        )

        # Measurement operator
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Run the quantum circuit on a batch of states and produce logits.

        Parameters
        ----------
        state_batch : torch.Tensor
            Tensor of shape (batch_size, 2**num_features) containing
            complex amplitudes of the input states.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, 2).
        """
        batch = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.num_features,
            bsz=batch,
            device=state_batch.device,
        )
        # Encode the classical data
        self.encoder(qdev, state_batch)

        # Apply the variational layers
        self.var_layer(qdev)

        # Measure in the Z basis
        meas = self.measure(qdev)

        # Feed the expectation values through the head
        return self.head(meas).squeeze(-1)

    def var_layer(self, qdev: tq.QuantumDevice) -> None:
        """Variational layer that mirrors the Qiskit circuit."""
        for i, w in enumerate(self.weights):
            tqf.ry(qdev, wires=[i % self.num_features], params=w)
        for i in range(self.num_features - 1):
            tqf.cz(qdev, wires=[i, i + 1])


def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[HybridClassifierModel, Iterable[int], Iterable[int], List[int]]:
    """Construct a quantum hybrid classifier returning the expected tuple."""
    model = HybridClassifierModel(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = list(model.weight_sizes)
    observables = list(range(2))  # placeholder to match the API
    return model, encoding, weight_sizes, observables
