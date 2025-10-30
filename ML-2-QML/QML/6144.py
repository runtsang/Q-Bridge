"""Hybrid estimator implemented on Qiskit.

HybridEstimatorQNN builds a parameterised quantum circuit that mirrors the
photonic fraud‑detection layers.  The circuit is compiled into a
StatevectorEstimator which can be queried to obtain expectation values
for a given input vector.  The returned expectation can be fed into a
classical post‑processing network.

The scaling paradigm is *combination*: the circuit parameters are clipped
as in the photonic model, and a quantum estimator is used in tandem with
a classical neural network.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from dataclasses import dataclass
from typing import Iterable, Optional

@dataclass
class FraudLayerParameters:
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

def _apply_layer(q: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter (approximated with Ry rotations)
    q.ry(params.bs_theta, 0)
    q.ry(params.bs_phi, 1)
    # Phase shifters
    for i, phase in enumerate(params.phases):
        q.rz(phase, i)
    # Squeezing (approximated by Ry and Rz)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_eff = _clip(r, 5.0) if clip else r
        q.ry(r_eff, i)
        q.rz(phi, i)
    # Displacement (simulated with Ry)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_eff = _clip(r, 5.0) if clip else r
        q.ry(r_eff, i)
        q.rz(phi, i)
    # Kerr (approximated by Rz)
    for i, k in enumerate(params.kerr):
        k_eff = _clip(k, 1.0) if clip else k
        q.rz(k_eff, i)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    q = QuantumCircuit(2)
    _apply_layer(q, input_params, clip=False)
    for layer in layers:
        _apply_layer(q, layer, clip=True)
    return q

class HybridEstimatorQNN:
    """
    Quantum implementation of the hybrid estimator.  It constructs a
    Qiskit EstimatorQNN that uses a StatevectorEstimator backend.
    The circuit can be interrogated via `evaluate` to obtain the
    expectation value for a batch of inputs.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        estimator: Optional[StatevectorEstimator] = None,
    ) -> None:
        self.circuit = build_fraud_detection_program(input_params, layers)
        self.observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])
        self.estimator = estimator or StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            estimator=self.estimator,
        )

    def evaluate(self, inputs: list[list[float]]) -> list[float]:
        """Return expectation values for each input vector."""
        return self.estimator_qnn.evaluate(inputs)

    def __call__(self, inputs: list[list[float]]) -> list[float]:
        return self.evaluate(inputs)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "HybridEstimatorQNN"]
