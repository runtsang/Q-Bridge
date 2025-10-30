"""Quantum implementation of the hybrid fraud detection model.

The code builds a parameterized 2‑qubit circuit that encodes the two input features via Ry gates,
interleaves them with trainable CNOT‑plus‑Ry layers, and samples the final state with a
`SamplerQNN`.  The sampler outputs a probability vector over two measurement outcomes,
which is then linearly mapped to the desired number of classes.  This mirrors the
classical architecture defined in the corresponding ML module, enabling a direct
comparison of performance between the two paradigms.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudDetectionHybridModel(nn.Module):
    """
    Hybrid quantum‑classical fraud detection model.

    The quantum part encodes the inputs into a 2‑qubit state, applies a
    variational layer with trainable Ry rotations, and samples the
    resulting state.  The sampler output is post‑processed by a
    small classical linear head to produce logits for fraud / non‑fraud.
    """
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # Parameter vectors for inputs and trainable weights
        self.input_params = ParameterVector("x", 2)
        self.weight_params = ParameterVector("w", 4)

        # Build the base circuit
        self.base_circuit = QuantumCircuit(2)
        # Input encoding
        self.base_circuit.ry(self.input_params[0], 0)
        self.base_circuit.ry(self.input_params[1], 1)
        # Variational layer (two Ry gates + a CNOT)
        self.base_circuit.cx(0, 1)
        for i in range(4):
            self.base_circuit.ry(self.weight_params[i], i % 2)

        # Sampler primitive
        self.simulator = AerSimulator(method="aer_simulator_statevector")
        self.sampler = SamplerQNN(
            circuit=self.base_circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.simulator,
        )

        # Classical post‑processing head
        self.classifier = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert inputs to numpy for the sampler
        inputs_np = x.detach().cpu().numpy()
        batch = inputs_np.shape[0]
        probs = []
        for b in range(batch):
            prob = self.sampler(inputs_np[b])
            probs.append(prob)
        probs_np = np.stack(probs, axis=0)
        probs_tensor = torch.from_numpy(probs_np).to(x.device).float()
        logits = self.classifier(probs_tensor)
        return logits


__all__ = ["FraudDetectionHybridModel"]
