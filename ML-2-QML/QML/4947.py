"""Quantum module that implements a hybrid quantum neural network:
- A 4‑wire encoder that embeds classical features.
- A random layer with 50 gates to enhance expressibility.
- A quantum convolution submodule that applies a random 2‑qubit gate per patch.
- Measurement of all qubits in the Z basis.
The module is compatible with torchquantum and can be used as a drop‑in replacement
for the classical QFCModel part of HybridNAT.
"""

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
import qiskit as qk
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
import networkx as nx
import numpy as np


class HybridQuantumModule(tq.QuantumModule):
    """Quantum feature extractor with a convolution‑like submodule."""

    class QConvLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            # Random 2‑qubit gates for each pair of consecutive qubits
            self.gates = [tq.RandomLayer(n_ops=10, wires=[i, i + 1]) for i in range(n_qubits - 1)]

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for gate in self.gates:
                gate(qdev)

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a single scalar feature into a 4‑qubit state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Quantum convolution layer
        self.qconv = self.QConvLayer(self.n_wires)
        # Main quantum layer
        self.q_layer = self.QLayer(self.n_wires)
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum module.

        Parameters
        ----------
        x : torch.Tensor
            Batch of scalar features of shape (B, 1).

        Returns
        -------
        torch.Tensor
            Normalized measurement results of shape (B, 4).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode each scalar into a 4‑qubit state
        self.encoder(qdev, x)
        # Apply quantum convolution
        self.qconv(qdev)
        # Apply main quantum layer
        self.q_layer(qdev)
        # Measure all qubits
        out = self.measure(qdev)
        return self.norm(out)


class HybridQuantumEstimator:
    """Simple Qiskit estimator that mirrors EstimatorQNN but uses a quantum circuit."""

    def __init__(self):
        # Define a tiny parameterised circuit
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.rx(theta, 0)
        # Observable: Pauli Y
        observable = qk.quantum_info.SparsePauliOp.from_list([("Y", 1)])
        # Wrap with Qiskit ML estimator
        self.estimator = Estimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[theta],
            weight_params=[theta],
            estimator=self.estimator
        )

    def predict(self, x):
        """Predict using the Qiskit EstimatorQNN."""
        return self.estimator_qnn.predict(x)


__all__ = ["HybridQuantumModule", "HybridQuantumEstimator"]
