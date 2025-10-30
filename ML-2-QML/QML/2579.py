from __future__ import annotations

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from dataclasses import dataclass
from typing import Iterable

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

def _apply_photonic_layer(modes: Iterable[int], params: FraudLayerParameters, clip: bool) -> None:
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for phase in params.phases:
        Rgate(phase) | modes[0]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for phase in params.phases:
        Rgate(phase) | modes[1]

def QCNN() -> EstimatorQNN:
    conv_params = [
        FraudLayerParameters(
            bs_theta=0.1, bs_phi=0.2,
            phases=(0.3, 0.4),
            squeeze_r=(0.5, 0.6),
            squeeze_phi=(0.7, 0.8),
            displacement_r=(0.9, 1.0),
            displacement_phi=(1.1, 1.2),
            kerr=(0.0, 0.0),
        ),
        FraudLayerParameters(
            bs_theta=0.15, bs_phi=0.25,
            phases=(0.35, 0.45),
            squeeze_r=(0.55, 0.65),
            squeeze_phi=(0.75, 0.85),
            displacement_r=(0.95, 1.05),
            displacement_phi=(1.15, 1.25),
            kerr=(0.0, 0.0),
        ),
    ]
    pool_params = [
        FraudLayerParameters(
            bs_theta=0.2, bs_phi=0.3,
            phases=(0.4, 0.5),
            squeeze_r=(0.6, 0.7),
            squeeze_phi=(0.8, 0.9),
            displacement_r=(1.0, 1.1),
            displacement_phi=(1.2, 1.3),
            kerr=(0.0, 0.0),
        ),
        FraudLayerParameters(
            bs_theta=0.25, bs_phi=0.35,
            phases=(0.45, 0.55),
            squeeze_r=(0.65, 0.75),
            squeeze_phi=(0.85, 0.95),
            displacement_r=(1.05, 1.15),
            displacement_phi=(1.25, 1.35),
            kerr=(0.0, 0.0),
        ),
    ]

    feature_map = ZFeatureMap(8)
    ansatz = sf.Program(8)
    with ansatz.context as q:
        # Convolution 1
        _apply_photonic_layer([q[0], q[1]], conv_params[0], clip=False)
        _apply_photonic_layer([q[2], q[3]], conv_params[0], clip=False)
        _apply_photonic_layer([q[4], q[5]], conv_params[0], clip=False)
        _apply_photonic_layer([q[6], q[7]], conv_params[0], clip=False)
        # Pooling 1
        _apply_photonic_layer([q[0], q[1]], pool_params[0], clip=True)
        _apply_photonic_layer([q[2], q[3]], pool_params[0], clip=True)
        _apply_photonic_layer([q[4], q[5]], pool_params[0], clip=True)
        _apply_photonic_layer([q[6], q[7]], pool_params[0], clip=True)
        # Convolution 2
        _apply_photonic_layer([q[0], q[1]], conv_params[1], clip=False)
        _apply_photonic_layer([q[2], q[3]], conv_params[1], clip=False)
        # Pooling 2
        _apply_photonic_layer([q[0], q[1]], pool_params[1], clip=True)
        _apply_photonic_layer([q[2], q[3]], pool_params[1], clip=True)
        # Convolution 3
        _apply_photonic_layer([q[0], q[1]], conv_params[0], clip=False)
        # Pooling 3
        _apply_photonic_layer([q[0], q[1]], pool_params[0], clip=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=ansatz,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=[],
        estimator=estimator,
    )
    return qnn

__all__ = ["FraudLayerParameters", "QCNN"]
