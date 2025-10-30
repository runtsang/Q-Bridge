"""Quantum sampler network with a classical pre‑processing filter.

The quantum part is a parameterised sampler based on Qiskit’s
SamplerQNN.  It receives a 2‑dimensional feature vector extracted
from a 28×28 image by a 2×2 convolution (the same filter that is
used in the classical counterpart).  The circuit contains two
input parameters, four trainable weight parameters and a random
layer that injects additional expressivity.  Probabilities are
obtained by sampling the statevector.

This module aims to be a direct quantum replacement for the
HybridSamplerQNN defined in the classical module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler as QiskitSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


class HybridSamplerQNN(nn.Module):
    """
    Quantum hybrid sampler network.

    Architecture
    ------------
    * Classical 2×2 convolution (stride 2) to extract 4 feature maps
      from a 28×28 image.
    * Two of the resulting 784‑dimensional features are selected as
      input parameters for a 2‑qubit variational circuit.
    * The circuit contains a random layer (8 two‑qubit gates) and
      four trainable rotation angles.
    * Sampling is performed with Qiskit’s StatevectorSampler.
    * Outputs a probability distribution over two classes.

    Parameters
    ----------
    device : str
        Backend device for the Qiskit sampler (e.g.'statevector').
    """

    def __init__(self, device: str = "statevector") -> None:
        super().__init__()
        # Classical pre‑processing: same as in the classical module
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=2,
            stride=2,
        )
        self._build_quantum_circuit()
        self.sampler = QiskitSampler(name=device)

    def _build_quantum_circuit(self) -> None:
        # Two input parameters, four weight parameters
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        # Encode inputs
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        # Entanglement
        qc.cx(0, 1)
        # Variational layers
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        # Random layer for expressivity
        for _ in range(3):
            qc.ry(0.5, 0)
            qc.ry(0.3, 1)
            qc.cx(0, 1)

        self.circuit = qc
        self.input_params = inputs
        self.weight_params = weights

        # Wrap into Qiskit’s SamplerQNN
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Probability distribution over two classes.
        """
        # Classical pre‑processing
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        # Select two features as input parameters for the quantum circuit
        # For simplicity, we take the first two features.
        input_vals = x[:, :2].detach().cpu().numpy()

        # Run the sampler
        probs = []
        batch_size = 32
        for i in range(0, input_vals.shape[0], batch_size):
            batch_inputs = input_vals[i : i + batch_size]
            result = self.sampler_qnn(batch_inputs)
            probs.append(result)

        probs = torch.cat([torch.tensor(p, dtype=torch.float32) for p in probs], dim=0)
        return probs


__all__ = ["HybridSamplerQNN"]
