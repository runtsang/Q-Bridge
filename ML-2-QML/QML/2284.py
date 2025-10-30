"""Hybrid quantum sampler and quanvolution network.

The quantum version mirrors the classical architecture but replaces
the sampler with a parameterized QNN and the quanvolution filter
with a torchquantum module.  The sampler circuit outputs a
probability distribution that is used as rotation angles for the
quanvolution encoder, allowing the quantum sampler to influence the
feature extraction stage.

The scaling paradigm is *combination*:  the quantum sampler and the
quantum quanvolution filter share parameters that are jointly
optimized, demonstrating a hybrid quantum‑classical training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN


class HybridSamplerQuanvolution(nn.Module):
    """
    Quantum hybrid network that combines a QNN sampler with a
    torchquantum quanvolution filter.  The sampler circuit outputs a
    probability distribution over two angles that are fed into the
    encoder of the quanvolution filter.  The resulting quantum
    measurements are flattened and passed through a linear head.
    """

    def __init__(
        self,
        sampler_input_dim: int = 2,
        sampler_hidden_dim: int = 4,
        sampler_output_dim: int = 2,
        quanvolution_wires: int = 4,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        # --- Quantum sampler circuit ---
        inputs = ParameterVector("input", sampler_input_dim)
        weights = ParameterVector("weight", sampler_output_dim)

        qc = QuantumCircuit(sampler_input_dim)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)

        # Wrap in Qiskit ML SamplerQNN
        sampler = StatevectorSampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

        # --- Quantum quanvolution filter ---
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(quanvolution_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classifier head
        self.classifier = nn.Linear(quanvolution_wires * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor, sampler_inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, 28, 28).
        sampler_inputs : torch.Tensor
            Parameters for the sampler QNN of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        # 1. Obtain sampler probabilities from the QNN
        probs = self.sampler_qnn(sampler_inputs)  # (batch, 2)

        # 2. Prepare quantum device for quanvolution
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.encoder.n_wires, bsz=bsz, device=device)

        # 3. Encode image patches and sampler angles
        x_img = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # 4‑element patch
                patch = torch.stack(
                    [
                        x_img[:, r, c],
                        x_img[:, r, c + 1],
                        x_img[:, r + 1, c],
                        x_img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )  # (batch, 4)

                # Encode patch
                self.encoder(qdev, patch)

                # Modulate with sampler probabilities
                qdev.ry(probs[:, 0], 0)
                qdev.ry(probs[:, 1], 1)

                # Apply random layer and measurement
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.encoder.n_wires))

        # 4. Concatenate all patch measurements
        features = torch.cat(patches, dim=1)  # (batch, 4*14*14)

        # 5. Classify
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridSamplerQuanvolution"]
