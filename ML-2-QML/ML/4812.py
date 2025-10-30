"""Hybrid fraud‑detection model – classical implementation.

This module implements the classical side of the hybrid FraudDetection‑Hybrid model.
It builds a torch ``nn.Sequential`` encoder that mirrors the photonic circuit
used in the original ``FraudDetection.py`` seed, then optionally
provides a thin wrapper to generate the corresponding quantum circuit
using the Qiskit implementation in the companion ``FraudDetection__gen266_qml.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Parameter container – identical to the photonic version for consistency.
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clip a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single classical layer from photonic parameters."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_classical_encoder(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential PyTorch model mirroring the layered structure."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Main hybrid class – classical side
# --------------------------------------------------------------------------- #
class FraudDetectionHybridModel(nn.Module):
    """
    Hybrid fraud‑detection model with a classical encoder and a quantum
    classification head.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first photonic‑inspired layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers.
    quantum_interface : Optional[callable]
        Callable that returns a Qiskit circuit given a parameter dictionary.
        If ``None`` the class behaves purely classically.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        quantum_interface: Optional[callable] = None,
    ) -> None:
        super().__init__()
        self.encoder = build_classical_encoder(input_params, layers)
        self.quantum_interface = quantum_interface

    # --------------------------------------------------------------------- #
    # Classical inference
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the classical encoder."""
        return self.encoder(x)

    # --------------------------------------------------------------------- #
    # Optional quantum bridge
    # --------------------------------------------------------------------- #
    def _build_quantum_circuit(self) -> "QiskitCircuit":
        """Instantiate a quantum circuit via the supplied interface."""
        if self.quantum_interface is None:
            raise RuntimeError("Quantum interface is not configured.")
        return self.quantum_interface()

    def run_quantum(
        self,
        x: torch.Tensor,
        backend: str = "qasm_simulator",
        shots: int = 1024,
    ) -> torch.Tensor:
        """
        Evaluate the hybrid model using a Qiskit backend.

        The classical encoder is first applied to the data, producing a
        two‑dimensional embedding that is then fed as parameters into the
        quantum circuit.  The circuit is executed on the chosen backend and
        the expectation value of the Z observable on each qubit is returned.

        Parameters
        ----------
        x : torch.Tensor
            Raw input features of shape ``(batch, 2)``.
        backend : str
            Name of the Qiskit backend to execute the circuit.
        shots : int
            Number of shots for the back‑end simulation.

        Returns
        -------
        torch.Tensor
            Log‑probabilities for each class.
        """
        # Encode data classically first
        embedding = self.encoder(x).detach().numpy()
        # Prepare the quantum circuit
        circuit = self._build_quantum_circuit()
        # Bind parameters – the interface is assumed to expose a ``bind_params`` method
        bound_circuit = circuit.bind_params(embedding)
        # Execute
        from qiskit import execute, Aer
        job = execute(bound_circuit, Aer.get_backend(backend), shots=shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        # Compute expectation of Z on each qubit
        expectations = _expectation_from_counts(counts, bound_circuit.num_qubits)
        return torch.tensor(expectations, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Helper to compute expectations from raw counts
# --------------------------------------------------------------------------- #
def _expectation_from_counts(counts: dict, num_qubits: int) -> List[float]:
    """Return the expectation value of Pauli‑Z on each qubit from measurement counts."""
    expectations = [0.0] * num_qubits
    total = sum(counts.values())
    for bitstring, freq in counts.items():
        for i in range(num_qubits):
            bit = int(bitstring[::-1][i])  # LSB first
            expectations[i] += (1 if bit == 0 else -1) * freq
    return [e / total for e in expectations]

__all__ = ["FraudDetectionHybridModel", "FraudLayerParameters", "build_classical_encoder"]
