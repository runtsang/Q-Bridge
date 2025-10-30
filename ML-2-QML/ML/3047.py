"""Hybrid Quanvolution-QCNN classifier combining classical and quantum modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from.Quanvolution__gen320_qml import QuanvolutionFilter, QCNN_qiskit


class QuanvolutionQCNNClassifier(nn.Module):
    """
    Hybrid model: classical quanvolution filter followed by a quantum QCNN.

    The quanvolution filter produces a 4×14×14 feature map (784 features).
    A linear projection reduces this to 8 dimensions, the input size required by the QCNN.
    The QCNN is constructed via the qiskit EstimatorQNN and outputs a single expectation value
    per sample, which is turned into class probabilities with LogSoftmax.
    """

    def __init__(self) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.projection = nn.Linear(4 * 14 * 14, 8)
        self.qcnn = QCNN_qiskit()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical quanvolution
        feat = self.quanvolution(x)          # shape: (batch, 784)
        # Project to 8‑dim quantum input
        quantum_input = self.projection(feat)  # shape: (batch, 8)
        # Convert to numpy for qiskit estimator
        quantum_input_np = quantum_input.detach().cpu().numpy()
        # Run QCNN via EstimatorQNN
        quantum_output = self.qcnn.predict(quantum_input_np)  # shape: (batch, 1)
        quantum_output_torch = torch.from_numpy(quantum_output).to(x.device)
        logits = quantum_output_torch.squeeze(-1)
        return self.log_softmax(logits)
