"""Hybrid estimator with a photonic‑inspired quantum circuit.

The quantum part is built with Qiskit and mirrors the classical fraud‑detection
layers.  Parameters are passed through a beam‑splitter‑style rotation
(`RZ` + `RX`), a phase gate (`RZ`), and a tunable `RY` to emulate squeezing
and displacement.  The circuit is wrapped in a Qiskit
`EstimatorQNN` so it can be trained with the same interface as a
classical neural network.

The module defines:

* :class:`FraudLayerParameters` – same hyper‑parameters as the classical
  counterpart.
* :func:`build_quantum_fraud_circuit` – constructs a parameterised
  two‑qubit circuit from the fraud parameters.
* :class:`HybridEstimatorQNN` – a Qiskit EstimatorQNN that uses the
  constructed circuit and a state‑vector estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic or classical layer."""
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


def build_quantum_fraud_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Construct a two‑qubit parameterised Qiskit circuit that emulates the
    photonic fraud‑detection layers.  The circuit is compatible with
    a `StatevectorEstimator` backend.

    Parameters are applied as follows:

    * `bs_theta` and `bs_phi` → `RZ` and `RX` on each qubit.
    * `phases` → second `RZ` per qubit.
    * `squeeze_r` → `RY` (squeezing analogue).
    * `displacement_r` → additional `RX` (displacement analogue).
    * `kerr` → `ZZ` interaction via a controlled‑Z with tunable angle.
    """
    qc = QuantumCircuit(2)
    # Input layer
    _apply_layer(qc, input_params, clip=False)
    # Hidden layers
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc


def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter analogue – rotations
    qc.rz(params.bs_phi, 0)
    qc.rx(params.bs_theta, 0)
    qc.rz(params.bs_phi, 1)
    qc.rx(params.bs_theta, 1)

    # Phase gates
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)

    # Squeezing analogue
    for i, r in enumerate(params.squeeze_r):
        qc.ry(_clip(r, 5.0) if clip else r, i)

    # Displacement analogue
    for i, r in enumerate(params.displacement_r):
        qc.rx(_clip(r, 5.0) if clip else r, i)

    # Kerr interaction analogue – controlled‑Z with tunable angle
    for i, k in enumerate(params.kerr):
        qc.cz(0, 1)
        qc.rz(_clip(k, 1.0) if clip else k, 0)
        qc.rz(-_clip(k, 1.0) if clip else -k, 1)
        qc.cz(0, 1)


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class HybridEstimatorQNN:
    """Qiskit EstimatorQNN that uses the photonic‑style circuit."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.qc = build_quantum_fraud_circuit(input_params, layers)
        # Observable: Y on qubit 0 (analogous to the original example)
        observable = SparsePauliOp.from_list([("Y" + "I" * (self.qc.num_qubits - 1), 1)])
        # State‑vector estimator backend
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.qc,
            observables=observable,
            input_params=[Parameter("input1")],
            weight_params=[Parameter("weight1")],
            estimator=estimator,
        )

    def predict(self, inputs: list[float]) -> list[float]:
        """Run the quantum circuit with given inputs and return expectation values."""
        return self.estimator_qnn.predict(inputs)

__all__ = [
    "FraudLayerParameters",
    "build_quantum_fraud_circuit",
    "HybridEstimatorQNN",
]
