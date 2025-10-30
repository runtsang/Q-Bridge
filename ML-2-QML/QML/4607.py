"""
Quantum implementation of a hybrid quanvolution network.
It uses TorchQuantum for a patch‑wise quantum kernel and a Qiskit
EstimatorQNN for quantum‑enabled regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum‑enabled quanvolution network.

    Parameters
    ----------
    mode : str, optional
        One of ``'classification'`` or ``'regression'``.  Determines the
        default head used in :meth:`forward`.
    n_classes : int, optional
        Number of classes for classification head.
    use_qiskit_estimator : bool, optional
        Whether to use a Qiskit EstimatorQNN for regression.
    """

    def __init__(
        self,
        mode: str = "classification",
        n_classes: int = 10,
        use_qiskit_estimator: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.use_qiskit_estimator = use_qiskit_estimator

        # 4‑wire quantum device for each 2×2 patch
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(self.n_wires, device="cpu")

        # Encoding of raw pixel values
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Random layer to generate entanglement
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical linear heads
        self.linear_cls = nn.Linear(4 * 14 * 14, n_classes)

        # Quantum estimator for regression
        if self.use_qiskit_estimator:
            # 1‑qubit circuit with a parameterized rotation
            self.input_param = tq.Parameter("input")
            self.weight_param = tq.Parameter("weight")
            qc = QuantumCircuit(1)
            qc.h(0)
            qc.ry(self.input_param, 0)
            qc.rx(self.weight_param, 0)
            observable = SparsePauliOp.from_list([("Y", 1)])
            estimator = Estimator()
            self.estimator_qnn = QiskitEstimatorQNN(
                circuit=qc,
                observable=observable,
                input_params=[self.input_param],
                weight_params=[self.weight_param],
                estimator=estimator,
            )
            # Trainable weight parameter
            self.reg_weight = nn.Parameter(torch.randn(1))

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, 1, 28, 28).
        mode : str, optional
            Overrides the default mode.  One of ``'classification'`` or
            ``'regression'``.
        """
        if mode is None:
            mode = self.mode

        bsz = x.shape[0]
        patches = []

        # Extract 2×2 patches and evaluate the quantum kernel
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Grab the four pixels
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(self.q_device, data)
                self.q_layer(self.q_device)
                measurement = self.measure(self.q_device)
                patches.append(measurement.view(bsz, 4))

        features = torch.cat(patches, dim=1)  # (B, 4*14*14)

        if mode == "classification":
            logits = self.linear_cls(features)
            return F.log_softmax(logits, dim=-1)

        elif mode == "regression":
            # Aggregate features into a single scalar per sample
            input_vals = features.mean(dim=1)  # (B,)
            # Prepare input vector for the Qiskit estimator
            input_tensor = torch.stack(
                [input_vals, self.reg_weight.expand_as(input_vals)], dim=1
            )
            # Use the Qiskit EstimatorQNN to compute expectation values
            preds = torch.tensor(
                self.estimator_qnn.predict(input_tensor.numpy()), dtype=torch.float32
            )
            return preds

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # --------------------------------------------------------------------- #
    # Quantum kernel matrix
    # --------------------------------------------------------------------- #
    def kernel_matrix(self, a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> torch.Tensor:
        """
        Compute the Gram matrix between two collections of samples using the
        patch‑wise quantum kernel.

        Parameters
        ----------
        a : Iterable[torch.Tensor]
            First collection of images.  Each element is a tensor of shape
            (1, 28, 28).
        b : Iterable[torch.Tensor]
            Second collection of images.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (len(a), len(b)).
        """
        def _kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode x, apply random layer, encode y with negative angles,
            # then measure overlap.
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            self.encoder(self.q_device, -y)
            self.q_layer(self.q_device)
            # Overlap is abs of inner product between states
            return torch.abs(self.q_device.states.view(-1)[0])

        mat = torch.empty(len(a), len(b))
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                mat[i, j] = _kernel(xi, yj)
        return mat

    # --------------------------------------------------------------------- #
    # Convenience helpers
    # --------------------------------------------------------------------- #
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Shortcut for classification."""
        return self.forward(x, mode="classification")

    def regress(self, x: torch.Tensor) -> torch.Tensor:
        """Shortcut for regression."""
        return self.forward(x, mode="regression")


__all__ = ["QuanvolutionHybrid"]
