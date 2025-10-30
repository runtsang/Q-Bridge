"""Quantum quanvolution model.

This module implements the quantum branch of the hybrid
`QuanvolutionEstimator`.  It uses TorchQuantum to apply a
parameterized two‑qubit circuit to each 2×2 image patch and
measure the Pauli‑Z expectation values.  The same class name
`QuanvolutionEstimator` is used so that the quantum and
classical implementations can be swapped transparently.
"""

import torch
import torchquantum as tq
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator


class _QuantumPatchEncoder(tq.QuantumModule):
    """Quantum encoder for a single 2×2 patch."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Qiskit EstimatorQNN – kept as a reference
        qc = QuantumCircuit(self.n_wires)
        x = Parameter("x")
        w = Parameter("w")
        qc.ry(x, 0)
        qc.rx(w, 0)
        observable = tq.PauliZ
        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[x],
            weight_params=[w],
            estimator=estimator,
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch
            Tensor of shape (batch, 4) with pixel values.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 4) containing the measured Z
            expectation values for each qubit.
        """
        bsz = patch.shape[0]
        device = patch.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, patch)
        self.q_layer(qdev)
        return self.measure(qdev)


class QuanvolutionEstimator(tq.QuantumModule):
    """Quantum version of the hybrid quanvolution model."""
    def __init__(self) -> None:
        super().__init__()
        self.quantum_patch = _QuantumPatchEncoder()
        self.linear = tq.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 10).
        """
        bsz = x.shape[0]
        patches = []
        img = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = img[:, r, c:r+2, c+2:c+4].reshape(bsz, 4)
                qfeat = self.quantum_patch(patch)
                patches.append(qfeat)
        qfeat = torch.cat(patches, dim=1)
        logits = self.linear(qfeat)
        return F.log_softmax(logits, dim=-1)
