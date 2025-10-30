"""Unified quantum classifier that mirrors the classical interface.

The quantum implementation is based on the Qiskit circuit from the
QuantumClassifierModel seed.  It accepts the same arguments as the
classical version and exposes identical helper methods, making it
drop‑in compatible for experimentation.

The circuit encodes the input features into rotation angles, applies
a depth‑controlled variational ansatz, and measures Pauli‑Z on each
qubit.  The measurement results can be post‑processed classically
to obtain logits.  Optionally a classical sigmoid gate is applied
to the measurement vector, mimicking the LSTM gate logic from the
quantum LSTM seed.

"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class UnifiedClassifier(nn.Module):
    """
    Quantum variational classifier that can be used as a drop‑in
    replacement for the classical model.  The interface matches
    the classical ``UnifiedClassifier``: ``forward`` returns
    log‑probabilities, and helper methods expose the encoding
    indices, parameter counts, and observables.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (== input dimensionality).
    depth : int
        Depth of the variational ansatz.
    use_gating : bool, optional
        If True, a classical sigmoid gate is applied to the
        measurement vector before the final linear layer.
    gating_hidden : int, optional
        Hidden size of the classical gating network.
    device : str or torch.device, optional
        Device for torch tensors.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        use_gating: bool = False,
        gating_hidden: int | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_gating = use_gating
        self.gating_hidden = gating_hidden or num_qubits
        self.device = device

        # Build the variational circuit
        self.circuit, self.encoding_params, self.var_params, self.observables = self._build_circuit()

        # Classical post‑processing layers
        self.head = nn.Linear(num_qubits, 2)
        self.weight_sizes = [self.head.weight.numel() + self.head.bias.numel()]

        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(num_qubits, self.gating_hidden),
                nn.Sigmoid(),
            )
            self.weight_sizes.append(
                self.gate[0].weight.numel() + self.gate[0].bias.numel()
            )

        self.to(device)

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Construct a variational circuit with data‑encoding and CZ entanglement."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data encoding
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        # Variational ansatz
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Observables – Z on each qubit
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        return qc, list(encoding), list(weights), observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass – in practice the circuit would be executed on a
        simulator or quantum backend.  For demonstration we use
        a classical expectation value approximation: replace the
        quantum circuit with a linear embedding of the input
        followed by the classical head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_qubits).

        Returns
        -------
        torch.Tensor
            Log‑probabilities of shape (batch, 2).
        """
        # Simulate measurement of Pauli‑Z (placeholder)
        z_measure = 2 * (x > 0.5).float() - 1.0  # +1/-1

        if self.use_gating:
            z_measure = self.gate(z_measure) * z_measure

        logits = self.head(z_measure)
        return F.log_softmax(logits, dim=1)

    # ------------------------------------------------------------------
    #  Helper methods – mirror the classical interface
    # ------------------------------------------------------------------
    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def get_encoding_params(self) -> List[ParameterVector]:
        """Return the encoding parameters."""
        return self.encoding_params

    def get_var_params(self) -> List[ParameterVector]:
        """Return the variational parameters."""
        return self.var_params

    def get_observables(self) -> List[SparsePauliOp]:
        """Return the measurement observables."""
        return self.observables

    def get_weight_sizes(self) -> List[int]:
        """Return the list of trainable parameter counts."""
        return self.weight_sizes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_qubits={self.num_qubits}, "
            f"depth={self.depth}, use_gating={self.use_gating})"
        )


__all__ = ["UnifiedClassifier"]
