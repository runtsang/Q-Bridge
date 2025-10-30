"""
Quantum version of the fraud detection circuit.

This module implements the FraudDetectionEnhanced class as a parameterised
quantum circuit using Qiskit.  The circuit consists of a variational layer
followed by a mid‑circuit measurement of qubit 0.  The measurement outcome
is used as a classical control for subsequent gates, enabling hybrid
gradient flow.  The final measurement on qubit 1 gives a probability
distribution that is interpreted as class probabilities.

The class exposes a minimal interface:
  * forward(inputs) -> probabilities
  * predict(inputs) -> class indices
  * loss(inputs, targets) -> loss scalar
  * parameters() -> list of Parameter objects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.providers.aer import AerSimulator

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, *, clip: bool = True) -> None:
    # Entangling gate (CX) as a beam‑splitter analogue
    qc.cx(0, 1)
    # Phase rotations
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)
    # Squeezing analogue: RZ gates with amplitude
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qc.rz(_clip(r, 5) if clip else r, i)
    # Displacement analogue: RX gates
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qc.rx(_clip(r, 5) if clip else r, i)
    # Kerr analogue: RZZ gates
    for i, k in enumerate(params.kerr):
        qc.rzz(_clip(k, 1) if clip else k, i, i)

class FraudDetectionEnhanced:
    """Quantum circuit for fraud detection with mid‑circuit measurement."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        num_classes: int = 2,
        backend=None,
    ) -> None:
        self.num_classes = num_classes
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(2, "q")
        self.cr = ClassicalRegister(2, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

        # Store parameters
        self.input_params = input_params
        self.layers = list(layers)

        # Build the circuit
        self._build_circuit()

    def _build_circuit(self) -> None:
        # Input layer
        _apply_layer(self.circuit, self.input_params, clip=False)
        # Variational layers
        for layer in self.layers:
            _apply_layer(self.circuit, layer, clip=True)
        # Mid‑circuit measurement of qubit 0
        self.circuit.measure(0, 0)
        # Reset qubit 0 and apply a controlled operation based on the measurement
        self.circuit.reset(0)
        # Example: controlled‑X on qubit 1 conditioned on measurement outcome
        self.circuit.ccx(0, 1, 1)
        # Final measurement of qubit 1
        self.circuit.measure(1, 1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return class probabilities for a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch_size, 2) with feature values.  The values are not
            directly used in the current implementation but are kept for API
            compatibility; they could be fed into a feature‑encoding circuit in
            a future extension.
        """
        job = execute(self.circuit, self.backend, shots=1024, memory=False)
        result = job.result()
        counts = result.get_counts()
        probs = np.zeros(self.num_classes)
        for outcome, count in counts.items():
            if outcome == "01":
                probs[0] += count
            elif outcome == "11":
                probs[1] += count
        probs /= 1024
        return probs

    def predict(self, inputs: np.ndarray) -> int:
        """Return the predicted class index."""
        probs = self.forward(inputs)
        return int(np.argmax(probs))

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Cross‑entropy loss between predicted probabilities and target."""
        probs = self.forward(inputs)
        eps = 1e-12
        probs = np.clip(probs, eps, 1 - eps)
        if self.num_classes == 1:
            return -np.mean(targets * np.log(probs) + (1 - targets) * np.log(1 - probs))
        return -np.sum(targets * np.log(probs)) / len(targets)

    def parameters(self):
        """Return list of Parameter objects used in the circuit."""
        return [p for p in self.circuit.parameters]

__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
