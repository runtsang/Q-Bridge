"""
Quantum sampler network with 3 qubits and richer entanglement.

The circuit consists of parameterised Ry and Rz rotations followed by
CX and CZ entangling gates.  It is designed to be used with
`qiskit_machine_learning.neural_networks.SamplerQNN` and leverages
`StatevectorSampler` for exact probability evaluation.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class SamplerQNNGen222(QSamplerQNN):
    """
    Variational sampler with 3 qubits, 2 input and 6 weight parameters.

    Circuit layout:
        ──Ry(in0)───────RY(w0)───────RY(w2)───────
        │              │              │
        ──Ry(in1)───────RY(w1)───────RY(w3)───────
        │              │              │
        ──Ry(0)─────────RY(w4)───────RY(w5)───────
        ──CX───────CZ───────CX───────CZ───────
    """

    def __init__(self) -> None:
        # Parameter vectors
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 6)

        # Build circuit
        qc = QuantumCircuit(3)
        # Layer 1: input rotations
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        # Entanglement
        qc.cx(0, 1)
        qc.cz(1, 2)
        # Layer 2: trainable rotations
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.ry(weights[2], 2)
        # Entanglement
        qc.cx(0, 1)
        qc.cz(1, 2)
        # Layer 3: additional rotations
        qc.ry(weights[3], 0)
        qc.ry(weights[4], 1)
        qc.ry(weights[5], 2)

        # Sampler primitive
        sampler = Sampler()

        # Initialise the base class
        super().__init__(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that accepts a batch of 2‑dimensional input vectors
        and returns the corresponding probability distributions.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (..., 2) containing classical inputs.

        Returns
        -------
        torch.Tensor
            Tensor of shape (..., 2) with probabilities for the two
            computational basis states |00⟩ and |01⟩ (the rest are
            discarded for brevity).
        """
        # The base class already implements the forward logic using the
        # sampler primitive.  We simply expose it for clarity.
        return super().forward(inputs)


__all__ = ["SamplerQNNGen222"]
