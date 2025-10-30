from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from dataclasses import dataclass
from typing import Iterable

@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer, re‑used for the quantum model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionHybrid:
    """
    Quantum fraud detection model that emulates the photonic circuit using
    parameterised gates and leverages Qiskit’s EstimatorQNN for prediction.
    """

    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.circuit = self._build_circuit()
        self.estimator_qnn = self._build_estimator_qnn()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        self._apply_layer(qc, self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(qc, layer, clip=True)
        return qc

    def _apply_layer(self,
                     qc: QuantumCircuit,
                     params: FraudLayerParameters,
                     *,
                     clip: bool) -> None:
        # Beam‑splitter → RZZ (approximate)
        qc.rzz(2 * params.bs_theta, 0, 1)
        qc.rzz(2 * params.bs_phi, 0, 1)

        # Phase shifts
        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)

        # Squeezing → RY rotations (approximate)
        for i, r in enumerate(params.squeeze_r):
            angle = self._clip(r, 5.0)
            qc.ry(angle, i)

        # Displacement → RX rotations (approximate)
        for i, r in enumerate(params.displacement_r):
            angle = self._clip(r, 5.0)
            qc.rx(angle, i)

        # Kerr → RZ rotations (approximate)
        for i, k in enumerate(params.kerr):
            angle = self._clip(k, 1.0)
            qc.rz(angle, i)

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _build_estimator_qnn(self) -> EstimatorQNN:
        # Observable: Y on first qubit
        observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])
        # Input parameters are the phases of the first qubit
        input_params = [Parameter("phase0")]
        # Weight parameters are the beam‑splitter angles of the first qubit
        weight_params = [Parameter("theta0"), Parameter("phi0")]
        estimator = StatevectorEstimator()
        return EstimatorQNN(circuit=self.circuit,
                            observables=observable,
                            input_params=input_params,
                            weight_params=weight_params,
                            estimator=ester)
    def evaluate(self, inputs: dict) -> float:
        """
        Evaluate the quantum model on a single input dictionary.
        The dictionary should map the input parameter names to values.
        """
        result = self.estimator_qnn.eval(inputs)
        return float(result[0].real)

__all__ = ["FraudDetectionHybrid"]
