"""Quantum implementation of HybridQuanvolutionEstimator using EstimatorQNN style circuit."""

from __future__ import annotations

import torch
import torchquantum as tq
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN


class HybridQuanvolutionEstimator(tq.QuantumModule):
    """
    Quantum-only variant of the hybrid model.  Builds a 4‑qubit circuit for each
    2×2 patch.  Each qubit receives a Ry gate driven by the pixel value
    (input parameter) and a trainable Ry gate (weight parameter).  The
    circuits are evaluated with a StatevectorEstimator and the expectation
    value of a Pauli‑Y observable is returned.  The resulting feature vector
    matches the shape produced by the classical counterpart.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.input_params = [Parameter(f"x_{i}") for i in range(self.n_wires)]
        self.weight_params = [Parameter(f"w_{i}") for i in range(self.n_wires)]
        self.circuit = QuantumCircuit(self.n_wires)
        for i in range(self.n_wires):
            self.circuit.h(i)
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[i], i)
        self.observable = SparsePauliOp.from_list([("Y" * self.n_wires, 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts a batch of images (B, 1, 28, 28).  For each 2×2 patch a
        quantum circuit is evaluated and its expectation value is
        collected.  The output is a flattened feature vector of shape
        (B, 4*14*14).
        """
        bsz, _, h, w = x.shape
        device = x.device
        features = []
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                patch = x[:, :, r:r+2, c:c+2]  # (B, 1, 2, 2)
                patch_values = patch.view(bsz, -1).detach().cpu().numpy()
                expectation = self.estimator_qnn(patch_values)
                expectation = torch.tensor(expectation, dtype=torch.float32, device=device)
                features.append(expectation)
        return torch.cat(features, dim=1)


__all__ = ["HybridQuanvolutionEstimator"]
