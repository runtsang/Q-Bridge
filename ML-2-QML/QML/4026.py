from __future__ import annotations

from dataclasses import dataclass
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Sampler

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class EstimatorQNNHybrid:
    """
    Quantum variational estimator that mirrors the fraud detection photonic
    circuit.  Parameters are mapped to a simple two‑qubit variational layer
    composed of RX/RY rotations and CX entangling gates.  The circuit is
    wrapped in a qiskit EstimatorQNN and can be used as a drop‑in
    replacement for the classical EstimatorQNN class.
    """
    def __init__(self, init_params: FraudLayerParameters) -> None:
        self.init_params = init_params
        self.estimator_qnn = self._build_estimator()

    def _build_estimator(self) -> EstimatorQNN:
        qc = QuantumCircuit(2)
        # Input encoding
        x = Parameter("x")
        qc.ry(x, 0)

        # Entangling block
        qc.cx(0, 1)

        # Weight parameters mapped from photonic design
        p = [Parameter(f"p{i}") for i in range(8)]
        values = [
            self.init_params.bs_theta,
            self.init_params.bs_phi,
            self.init_params.squeeze_r[0],
            self.init_params.squeeze_r[1],
            self.init_params.displacement_r[0],
            self.init_params.displacement_r[1],
            self.init_params.kerr[0],
            self.init_params.kerr[1],
        ]
        for param, val in zip(p, values):
            param.assign_value(_clip(val, 5.0))

        # Apply rotations
        qc.rx(p[0], 0); qc.ry(p[1], 0)
        qc.rx(p[2], 1); qc.ry(p[3], 1)
        qc.cx(0, 1)
        qc.rx(p[4], 0); qc.ry(p[5], 0)
        qc.rx(p[6], 1); qc.ry(p[7], 1)

        # Observable: Pauli Y on first qubit
        observable = Pauli("YI")
        estimator = Sampler()
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=[observable],
            input_params=[x],
            weight_params=p,
            estimator=estimator
        )
        return estimator_qnn

    def __call__(self, *args, **kwargs):
        """Delegate call to the underlying EstimatorQNN."""
        return self.estimator_qnn(*args, **kwargs)

__all__ = ["FraudLayerParameters", "EstimatorQNNHybrid"]
