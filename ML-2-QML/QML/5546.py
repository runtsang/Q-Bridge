"""Quantum estimator combining variational circuit, measurement, and sampler."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as SamplerPrimitive

class SamplerCircuit:
    """Builds a parameterized sampler circuit used by the quantum estimator."""
    def __init__(self, num_qubits: int = 2):
        self.inputs = ParameterVector("input", num_qubits)
        self.weights = ParameterVector("weight", 4)
        self.circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            self.circuit.ry(self.inputs[i], i)
        self.circuit.cx(0, 1)
        for w in self.weights[:2]:
            self.circuit.ry(w, 0)
        for w in self.weights[2:]:
            self.circuit.ry(w, 1)
        self.circuit.cx(0, 1)

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

class EstimatorQNNGen449(tq.QuantumModule):
    """Variational estimator that measures expectation values and samples."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

        # Sampler component
        sampler_obj = SamplerCircuit(num_qubits=2)
        self.sampler = QiskitSamplerQNN(
            circuit=sampler_obj.get_circuit(),
            input_params=sampler_obj.inputs,
            weight_params=sampler_obj.weights,
            sampler=SamplerPrimitive(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        features = self.measure(qdev)
        pred = self.head(features).squeeze(-1)
        # Run sampler to get probabilities (placeholder interface)
        probs = self.sampler(x)  # actual usage may differ
        return {"prediction": pred, "probabilities": probs}

__all__ = ["EstimatorQNNGen449"]
