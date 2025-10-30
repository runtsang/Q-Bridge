"""Quantum implementation of the hybrid classifier/regressor.

The class shares the public API of the classical version but performs
all heavy lifting on a quantum device.  It contains a small ansatz
layer and a GeneralEncoder that maps classical inputs into a
superposition state.  The final measurement feeds into a classical
linear head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from typing import Iterable, Tuple, List, Union


class HybridQuantumClassifier(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
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

    def __init__(
        self,
        num_wires: int,
        depth: int = 2,
        task: str = "classification",
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        self.task = task
        self.device = torch.device(device)

        # Encoder that maps classical data into a quantum state.
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )

        # Variational layer
        self.q_layer = self.QLayer(num_wires)

        # Measurement and classical head
        self.measure = tq.MeasureAll(tq.PauliZ)
        head_out = 2 if task == "classification" else 1
        self.head = nn.Linear(num_wires, head_out)

        # Metadata for compatibility
        self.encoding_indices: List[int] = list(range(num_wires))
        self.weight_sizes: List[int] = [p.numel() for p in self.parameters()]
        self.observables: List[tq.SparsePauliOp] = [
            tq.SparsePauliOp("I" * i + "Z" + "I" * (num_wires - i - 1))
            for i in range(num_wires)
        ]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Run the quantum circuit on a batch of classical states.
        `state_batch` should be a 2‑D tensor of shape (batch, num_wires)
        with values in [-π, π] which are treated as rotation angles.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.num_wires, bsz=bsz, device=self.device
        )
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    @staticmethod
    def generate_superposition_data(
        num_wires: int, samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate superposition states for regression tasks.
        The implementation follows the quantum regression seed.
        """
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels.astype(np.float32)

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[tq.QuantumCircuit, List[int], List[int], List[tq.SparsePauliOp]]:
        """
        Build a simple layered ansatz that mimics the classical circuit builder
        signature.  The returned tuple is identical to the one produced by the
        classical version, allowing the two implementations to be swapped
        at the API level.
        """
        encoding = list(range(num_qubits))
        weights = [f"theta_{i}" for i in range(num_qubits * depth)]
        circuit = tq.QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            circuit.rx(f"x_{q}", q)
        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                circuit.ry(f"theta_{idx}", q)
                idx += 1
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)
        observables = [
            tq.SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, encoding, weights, observables


__all__ = ["HybridQuantumClassifier"]
