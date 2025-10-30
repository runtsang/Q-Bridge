from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

# Quantum sampler inspired by the original SamplerQNN example
class FraudDetectionHybrid(nn.Module):
    """
    Quantum component that implements a parameterised 2‑qubit SamplerQNN.
    The forward pass accepts a batch of classical feature vectors and
    returns a probability vector for each sample. The circuit mirrors
    the architecture used in the reference SamplerQNN, but is wrapped
    as a PyTorch module for seamless integration.
    """

    def __init__(self, backend: str = "statevector_simulator") -> None:
        super().__init__()
        self.backend = Aer.get_backend(backend)
        self.circuit_template = self._build_circuit_template()
        self.weight_params = ParameterVector("weight", 4)

    def _build_circuit_template(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Placeholder; actual parameters will be bound during forward
        qc.rx(0, 0)
        qc.rx(0, 1)
        qc.cx(0, 1)
        qc.rx(0, 0)
        qc.rx(0, 1)
        qc.cx(0, 1)
        qc.rx(0, 0)
        qc.rx(0, 1)
        return qc

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, 2) – two rotation angles per sample for the first
            layer of the circuit.
        weights : torch.Tensor
            Shape (batch, 4) – four rotation angles for the trainable part
            of the circuit.

        Returns
        -------
        torch.Tensor
            Shape (batch, 8) – probability distribution over the 8 computational
            basis states of the 2‑qubit system.
        """
        batch_size = inputs.shape[0]
        probs = []

        for idx in range(batch_size):
            qc = self.circuit_template.copy()
            # Bind input rotations
            qc.ry(inputs[idx, 0].item(), 0)
            qc.ry(inputs[idx, 1].item(), 1)
            # Bind trainable weights
            qc.ry(weights[idx, 0].item(), 0)
            qc.ry(weights[idx, 1].item(), 1)
            qc.cx(0, 1)
            qc.ry(weights[idx, 2].item(), 0)
            qc.ry(weights[idx, 3].item(), 1)

            result = execute(qc, backend=self.backend).result()
            statevector = result.get_statevector()
            probability = np.abs(statevector) ** 2
            probs.append(probability)

        return torch.tensor(np.stack(probs), dtype=torch.float32)

    def sample_probabilities(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that converts the output of the forward pass
        into a torch tensor on the same device as the inputs.
        """
        return self.forward(inputs, weights).to(inputs.device)

__all__ = ["FraudDetectionHybrid"]
