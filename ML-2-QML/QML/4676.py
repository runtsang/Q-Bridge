"""Quantum regression model built on Qiskit EstimatorQNN.

The model maps each feature into a rotation on a dedicated qubit,
entangles all qubits with a parameterized layer, and measures
the expectation of Pauli‑Z on each qubit.  A classical linear head
produces the final scalar output.  The architecture mirrors the
classical model but leverages quantum feature maps.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Sampler
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import RealAmplitudes


def _build_circuit(num_qubits: int) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Create a parameterised circuit that encodes the input features.

    Each feature drives a Ry rotation on its qubit, followed by an
    entangling RealAmplitudes layer with trainable angles.
    """
    params = [Parameter(f"θ{i}") for i in range(num_qubits)]
    qc = QuantumCircuit(num_qubits)
    for i, p in enumerate(params):
        qc.ry(p, i)
    # Entangling layer
    entangler = RealAmplitudes(num_qubits, reps=2)
    qc.compose(entangler, inplace=True)
    return qc, params


class QModel(nn.Module):
    """Hybrid quantum–classical regression network."""

    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits

        # Quantum circuit and parameters
        self.circuit, self.input_params = _build_circuit(num_qubits)

        # Measurement observable – Pauli‑Z on all qubits
        self.observable = Pauli("Z" * num_qubits)

        # Classical head
        self.head = nn.Linear(num_qubits, 1)

        # Estimator primitive
        self.estimator = Sampler()

        # Build EstimatorQNN wrapper
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[self.observable],
            input_params=self.input_params,
            estimator=self.estimator,
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Run the batch through the quantum circuit and head.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, num_qubits) – each entry is used as a rotation angle.
        """
        batch_size = state_batch.shape[0]
        # Convert to numpy for EstimatorQNN (expects list of param values)
        param_values = state_batch.detach().cpu().numpy().tolist()

        # Execute the quantum circuit batch‑wise
        # The EstimatorQNN returns expectation values for each input
        expectations = self.estimator_qnn.evaluate(param_values).values
        expectations = torch.tensor(expectations, dtype=torch.float32, device=state_batch.device)

        # Classical linear head
        return self.head(expectations).squeeze(-1)


__all__ = ["QModel"]
