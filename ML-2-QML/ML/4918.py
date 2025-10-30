"""Hybrid sampler that combines a quantum kernel with a classical MLP."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


class QuantumKernelLayer(nn.Module):
    """A lightweight quantum kernel that evaluates a 2‑qubit circuit."""

    def __init__(self) -> None:
        super().__init__()
        self.num_qubits = 2
        self.weight_params = ParameterVector("theta", 4)

    def forward(self, x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        # x shape (batch, 2), parameters shape (batch, 4)
        batch = x.shape[0]
        features = torch.empty(batch, 2, device=x.device, dtype=torch.float32)
        for i in range(batch):
            qc = QuantumCircuit(self.num_qubits)
            # Encode classical data
            qc.ry(x[i, 0], 0)
            qc.ry(x[i, 1], 1)
            # Variational block
            qc.ry(parameters[i, 0], 0)
            qc.ry(parameters[i, 1], 1)
            qc.cx(0, 1)
            qc.ry(parameters[i, 2], 0)
            qc.ry(parameters[i, 3], 1)
            # Expectation values of Z on each qubit
            state = Statevector.from_instruction(qc)
            features[i, 0] = state.expectation_value('Z' * self.num_qubits)[0]
            features[i, 1] = state.expectation_value('Z' * self.num_qubits)[1]
        return features


class HybridSamplerModule(nn.Module):
    """Hybrid sampler: quantum kernel + classical MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel = QuantumKernelLayer()
        self.classifier = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2)
        )

    def forward(self,
                inputs: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Classical feature vector of shape (batch, 2).
        weights : torch.Tensor
            Variational parameters of shape (batch, 4).
        Returns
        -------
        torch.Tensor
            Softmax probabilities over two classes.
        """
        qfeat = self.kernel(inputs, weights)
        logits = self.classifier(qfeat)
        return F.softmax(logits, dim=-1)


class FastHybridEstimator:
    """Estimator that evaluates the hybrid model for batches of parameters."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: list[torch.Tensor],
        parameter_sets: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[list[float]]:
        """
        Parameters
        ----------
        observables : list[torch.Tensor]
            List of 1‑D tensors mapping from model output to a scalar.
        parameter_sets : list[tuple[np.ndarray, np.ndarray]]
            Each tuple contains (inputs, weights) arrays.
        Returns
        -------
        list[list[float]]
            Evaluation results for each observable and parameter set.
        """
        results: list[list[float]] = []
        self.model.eval()
        with torch.no_grad():
            for inp, wts in parameter_sets:
                inp_t = torch.as_tensor(inp, dtype=torch.float32)
                w_t = torch.as_tensor(wts, dtype=torch.float32)
                outputs = self.model(inp_t.unsqueeze(0), w_t.unsqueeze(0))
                row: list[float] = []
                for obs in observables:
                    val = obs(outputs)
                    row.append(float(val.squeeze()))
                results.append(row)
        return results


__all__ = ["HybridSamplerModule", "FastHybridEstimator"]
