"""Hybrid fraud detection module combining classical and quantum feature extraction."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

# Quantum imports are optional for the classical side; they are used only for simulation.
try:
    from qiskit import Aer, execute
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import RandomCircuit
except ImportError:
    # If Qiskit is not available, the quantum part will be a stub.
    Aer = execute = ParameterVector = RandomCircuit = None


class QuantumFeatureExtractor(nn.Module):
    """
    A light‑weight quantum feature extractor that simulates a parameterized circuit
    and returns expectation values of Pauli‑Z on each qubit.  The circuit contains
    an input encoding layer followed by a few random unitary layers, mirroring the
    structure of the Quanvolution random layer but with a fixed number of qubits.
    """

    def __init__(self, num_qubits: int = 2, num_layers: int = 2, backend_name: str = "statevector_simulator") -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.backend_name = backend_name
        if Aer is None:
            raise RuntimeError("Qiskit must be installed to use QuantumFeatureExtractor.")
        self.backend = Aer.get_backend(self.backend_name)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, num_qubits).  Each row contains the angles for the
            Ry encoding gates applied to the corresponding qubit.

        Returns
        -------
        torch.Tensor
            Shape (batch, num_qubits).  Expectation values of Pauli‑Z on each qubit.
        """
        batch_size = inputs.size(0)
        results = []

        for i in range(batch_size):
            # Build a fresh circuit for each sample
            circ = self._build_circuit(inputs[i].cpu().numpy())
            job = execute(circ, self.backend, shots=1024)
            state = job.result().get_statevector(circ)
            # Compute expectation values of Z for each qubit
            exp_vals = []
            for q in range(self.num_qubits):
                # Z expectation = sum_{b} (-1)^{b_q} |state_b|^2
                idxs = [idx for idx, bit in enumerate(format(b, f"0{self.num_qubits}b")) if bit == "1"]
                # Use statevector to compute probability amplitude
                prob = sum(abs(state[idx]) ** 2 for idx in range(len(state)))
                # For a simple simulator, we approximate expectation by sampling
                exp = 0.0
                for idx, amp in enumerate(state):
                    bit = (idx >> q) & 1
                    exp += ((-1) ** bit) * abs(amp) ** 2
                exp_vals.append(exp)
            results.append(torch.tensor(exp_vals, dtype=torch.float32, device=inputs.device))

        return torch.stack(results, dim=0)

    def _build_circuit(self, input_angles: list[float]) -> "QuantumCircuit":
        """Construct a parameterized circuit for a single sample."""
        from qiskit import QuantumCircuit

        circ = QuantumCircuit(self.num_qubits)
        # Input encoding with Ry gates
        for q, angle in enumerate(input_angles):
            circ.ry(angle, q)

        # Random unitary layers
        for _ in range(self.num_layers):
            rand_circ = RandomCircuit(self.num_qubits, depth=2).decompose()
            circ.append(rand_circ, range(self.num_qubits))

        return circ


class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection network.  The quantum feature extractor precedes a
    classical feed‑forward head.  The architecture is inspired by EstimatorQNN
    (regression head) and SamplerQNN (softmax output), but the final layer is
    a single‑output regression suitable for fraud‑risk scoring.
    """

    def __init__(self, num_qubits: int = 2, num_layers: int = 2) -> None:
        super().__init__()
        self.quantum_extractor = QuantumFeatureExtractor(num_qubits, num_layers)
        # Classical head: two hidden layers with Tanh activations
        self.classical_head = nn.Sequential(
            nn.Linear(num_qubits, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),  # Regression output
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, num_qubits).  Input features are encoded as angles for
            the Ry gates in the quantum circuit.

        Returns
        -------
        torch.Tensor
            Shape (batch, 1).  Fraud‑risk score between -∞ and ∞.
        """
        q_features = self.quantum_extractor(inputs)
        return self.classical_head(q_features)


__all__ = ["FraudDetectionHybrid"]
