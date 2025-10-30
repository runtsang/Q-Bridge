"""Hybrid classical‑quantum convolutional layer.

This module implements a drop‑in replacement for the original Conv filter. It
combines a standard Conv2d layer with a parameter‑shift differentiable
variational circuit. The `use_quantum` flag toggles the quantum branch.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import Parameter, QuantumCircuit
from typing import List

class ConvHybrid(nn.Module):
    """
    Drop‑in replacement for the original Conv filter that supports a hybrid
    classical‑quantum architecture.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_quantum: bool = True,
        n_layers: int = 2,
        shots: int = 100,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.n_layers = n_layers
        self.shots = shots

        # Classical branch
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=kernel_size, bias=True
        )
        self.sigmoid = nn.Sigmoid()

        # Quantum branch
        self.n_qubits = kernel_size ** 2
        # Torch parameter for learning
        self.theta = nn.Parameter(torch.randn(self.n_qubits, dtype=torch.float32))
        # Qiskit parameters for circuit
        self.q_params = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        # Build a reusable circuit template
        self.circuit_template = self._build_circuit_template()
        # Backend for simulation
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit_template(self) -> QuantumCircuit:
        """
        Return a circuit containing all parameterised gates but no data encoding.
        """
        qc = QuantumCircuit(self.n_qubits)
        # Parameterised rotations
        for i in range(self.n_qubits):
            qc.ry(self.q_params[i], i)
        # Entangling layer: CX between adjacent qubits (linear chain)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        # Repeat to reach desired depth
        for _ in range(self.n_layers - 1):
            for i in range(self.n_qubits):
                qc.ry(self.q_params[i], i)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Expects input shape (batch, 1, H, W).
        Returns a scalar per image (batch, 1).
        """
        # Classical path
        logits = self.conv(x)
        logits = self.sigmoid(logits)
        # Mean of the single‑channel output
        classical_out = logits.mean(dim=(2, 3)).unsqueeze(-1)  # shape (batch,1)

        if not self.use_quantum:
            return classical_out

        # Quantum path
        batch_size = x.shape[0]
        # Flatten each image to a vector of length n_qubits
        patches = x.view(batch_size, -1).detach().cpu().numpy()

        # Encode data: thresholding -> pi or 0
        encodings = (patches > self.threshold).astype(np.float32) * np.pi

        # Prepare a list of circuits with data‑dependent bindings
        circuits: List[QuantumCircuit] = []
        for i in range(batch_size):
            qc = self.circuit_template.copy()
            # Bind the learnable parameters
            bind_dict = {self.q_params[j]: self.theta[j].item() for j in range(self.n_qubits)}
            # Bind data‑specific parameters
            for j in range(self.n_qubits):
                bind_dict[self.q_params[j]] = encodings[i, j]
            qc.bind_parameters(bind_dict)
            qc.measure_all()
            circuits.append(qc)

        # Execute all circuits in a single job
        job = execute(circuits, self.backend, shots=self.shots)
        result = job.result()

        # Compute expectation value of Z for each qubit and average
        quantum_vals = []
        for circ in circuits:
            counts = result.get_counts(circ)
            # Convert counts to expectation of Z: +1 for |0>, -1 for |1>
            exp = 0.0
            for bitstring, count in counts.items():
                # bitstring is in reverse order; reverse it for consistency
                bits = bitstring[::-1]
                z_sum = sum(1 if b == "0" else -1 for b in bits)
                exp += z_sum * count
            exp = exp / (self.shots * self.n_qubits)
            quantum_vals.append(exp)

        quantum_out = torch.tensor(quantum_vals, dtype=torch.float32, device=x.device).unsqueeze(-1)

        return classical_out + quantum_out
