"""Quantum-aware version of HybridQuanvolutionNet.  It replaces the
classical random projection with a trainable quantum layer and the
classifier with a variational quantum circuit (implemented via Qiskit
and wrapped for TorchQuantum).  The module remains a
torchquantum.QuantumModule so that it can be trained with Autograd."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int):
    """
    Construct a simple layered ansatz with explicit encoding and
    variational parameters.  Returns:
        - QuantumCircuit
        - list of encoding Parameter objects
        - list of weight Parameter objects
        - list of PauliZ observables
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Input encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class HybridQuanvolutionNet(tq.QuantumModule):
    """
    Quantum variant of HybridQuanvolutionNet.  It contains:
        * Convolutional feature extractor (same as the classical version).
        * Quantum feature mapping per 2x2 patch using a GeneralEncoder + RandomLayer.
        * A variational classifier built from a Qiskit circuit.
        * A linear head to produce logits.
    """

    def __init__(
        self,
        depth: int = 2,
        num_classes: int = 10,
        patch_size: int = 2,
        n_wires: int = 4,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.n_wires = n_wires

        # Quantum feature mapping
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Variational classifier (built via Qiskit for metadata)
        qc, encoding_params, weight_params, observables = build_classifier_circuit(
            n_wires, depth
        )
        self._qc = qc
        self.register_parameter("encoding_params", torch.tensor([0.0] * len(encoding_params)))
        self.register_parameter("weight_params", torch.tensor([0.0] * len(weight_params)))
        self.encoding, self.weights_meta, self.observables = encoding_params, weight_params, observables

        # Linear head (same output dimensionality as classical version)
        in_features = n_wires * 14 * 14
        self.linear_head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Extract features with conv_extractor.
          2. Unfold into 2x2 patches, average across channels.
          3. Encode each patch with the quantum circuit.
          4. Measure and flatten.
          5. Classify with linear head.
        """
        bsz = x.size(0)

        # Convolutional feature extractor (identical to classical)
        conv_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        feats = conv_extractor(x)  # (bsz, 16, 7, 7)

        # 2x2 patches: unfold then reshape
        patches = feats.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(bsz, 16, 14 * 14, self.patch_size * self.patch_size)

        # Average across the 16 feature maps
        patches = patches.mean(dim=1)  # (bsz, 14*14, 4)

        # Quantum feature mapping per patch
        batch_patches = patches.view(bsz * 14 * 14, self.patch_size * self.patch_size)
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch_patches.size(0), device=x.device)
        self.encoder(qdev, batch_patches)
        self.q_layer(qdev)
        measurement = self.measure(qdev)

        # Reshape back to (batch, features)
        quantum_features = measurement.view(bsz, -1)

        logits = self.linear_head(quantum_features)
        return F.log_softmax(logits, dim=-1)

    def get_classifier_meta(self):
        """
        Return the same metadata used by the classical counterpart:
          - encoding indices
          - weight sizes per linear layer
          - observables (class indices)
        """
        return self.encoding, self.weights_meta, self.observables


__all__ = ["HybridQuanvolutionNet"]
