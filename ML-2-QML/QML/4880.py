"""Hybrid layer that fuses classical feature extraction with a variational quantum circuit.

The classical backbone is a shallow neural net inspired by QCNN and EstimatorQNN.  The
quantum block is a single‑qubit variational circuit whose rotation angle is generated
by the classical network.  The expectation value of the Pauli‑Y operator modulates
the class probabilities, enabling hybrid training on simulators or real hardware.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import Aer, execute, QuantumCircuit
import qiskit

class UnifiedHybridLayer(nn.Module):
    """Hybrid layer combining classical and quantum modules."""
    def __init__(self, in_features: int = 2, hidden: int = 8, num_classes: int = 2) -> None:
        super().__init__()
        # Classical feature extractor – shallow network
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(hidden // 2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

        # Quantum block – single‑qubit variational circuit
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

        # Combiner to fuse classical output and quantum expectation
        self.combiner = nn.Linear(num_classes + 1, num_classes)

    def quantum_expectation(self, theta_val: float) -> float:
        """Execute the parameterised circuit and return the Pauli‑Y expectation."""
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta_val}],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        bits = np.array([int(b) for b in counts.keys()])
        expectation = np.sum(bits * probs)
        return expectation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical forward pass
        feat = self.feature_extractor(x)
        class_out = self.classifier(feat)  # [batch, num_classes]

        # Generate quantum parameters from the first classical output
        theta_vals = class_out[:, 0].detach().cpu().numpy()

        # Compute quantum expectations for each sample
        expectations = [self.quantum_expectation(theta) for theta in theta_vals]
        quantum_out = torch.tensor(expectations, dtype=x.dtype, device=x.device).unsqueeze(-1)

        # Fuse classical and quantum outputs
        combined = torch.cat([class_out, quantum_out], dim=-1)
        out = self.combiner(combined)
        return self.softmax(out)
