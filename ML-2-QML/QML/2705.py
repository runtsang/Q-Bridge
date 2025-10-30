"""Quantum implementation of the fraud‑detection hybrid model.

The quantum circuit is a two‑qubit parameterised routine that mirrors the
photonic layer structure: beam‑splitter‑like entanglement, single‑mode
rotations (phases), squeezers (mapped to RZ), displacements (mapped to RX),
and Kerr non‑linearities (mapped to RZ with a larger angle).  The circuit
is wrapped in an EstimatorQNN so that the model can be trained end‑to‑end
with classical optimisers.

The class name matches the classical implementation so that the two
modules can be swapped in a research pipeline while keeping a shared
API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Dict, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer, reused for the quantum circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _build_quantum_layer(params: FraudLayerParameters, *, clip: bool) -> QuantumCircuit:
    """Return a 2‑qubit sub‑circuit implementing one photonic layer."""
    q = QuantumCircuit(2, name="layer")
    # Beam‑splitter analogue: CX
    q.cx(0, 1)
    # Single‑mode rotations (phases)
    for i, phase in enumerate(params.phases):
        q.rz(phase, i)
    # Squeezing analogue: RZ with bound
    for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        q.rz(_clip(r, 5) if clip else r, i)
    # Entangle again
    q.cx(0, 1)
    # Displacements analogue: RX
    for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        q.rx(_clip(r, 5) if clip else r, i)
    # Kerr analogue: RZ with bound
    for i, k in enumerate(params.kerr):
        q.rz(_clip(k, 1) if clip else k, i)
    return q

class FraudDetectionHybrid:
    """Quantum fraud‑detection hybrid model using EstimatorQNN."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        hidden_sizes: Sequence[int] = (8, 4),
    ) -> None:
        # Build the full circuit
        qc = QuantumCircuit(2)
        qc.append(_build_quantum_layer(input_params, clip=False), [0, 1])
        for l in layers:
            qc.append(_build_quantum_layer(l, clip=True), [0, 1])
        # Final linear mapping (two X gates)
        qc.x(0)
        qc.x(1)

        # Observable: Pauli Y on both qubits
        observable = SparsePauliOp.from_list([("YY", 1)])

        # Estimator backend
        estimator = StatevectorEstimator()

        # Build EstimatorQNN
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[Parameter("input1")],  # placeholder for data encoding
            weight_params=[Parameter(f"w{i}") for i in range(len(hidden_sizes))],
            estimator=estimator,
        )
        self.hidden_sizes = hidden_sizes

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return model predictions for a batch of 2‑dimensional inputs."""
        predictions = []
        for x in data:
            mapping: Dict[Any, float] = {
                self.estimator_qnn.input_params[0]: x[0],
                self.estimator_qnn.weight_params[0]: x[1],
            }
            # Remaining weight_params default to zero
            for wp in self.estimator_qnn.weight_params[1:]:
                mapping[wp] = 0.0
            predictions.append(self.estimator_qnn.predict(mapping)[0])
        return np.array(predictions)

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.estimator_qnn.circuit

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
