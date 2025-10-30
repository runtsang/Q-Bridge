"""Hybrid quantum regression model with encoder, random layer and optional sampler head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler


class QuantumRegressionModel(tq.QuantumModule):
    """Quantum encoder + random layer followed by a classical linear head.
    An optional sampler head is provided for probabilistic inference."""

    class QLayer(tq.QuantumModule):
        """Random variational layer operating on `num_wires` qubits."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.num_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.num_wires = num_wires
        # Classical‑style encoder mapping classical amplitudes to qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear head producing the regression output
        self.head = nn.Linear(num_wires, 1)
        # Optional sampler head for probabilistic output
        self.sampler = self._build_sampler(num_wires)

    def _build_sampler(self, num_wires: int):
        """Construct a Qiskit SamplerQNN that returns a probability distribution."""
        inputs = ParameterVector("input", num_wires)
        weights = ParameterVector("weight", 4)
        # Simple 2‑qubit variational circuit (extendable to more qubits)
        qc = tq.QuantumDevice(n_wires=num_wires)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        if num_wires > 2:
            qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        if num_wires > 2:
            qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        sampler = Sampler()
        return QiskitSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def sample(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Return sampled probabilities from the optional sampler head."""
        # Flatten batch into list of inputs
        inputs = state_batch.detach().cpu().numpy()
        # For simplicity, use the same classical features as circuit inputs
        return self.sampler(inputs)

__all__ = ["QuantumRegressionModel"]
