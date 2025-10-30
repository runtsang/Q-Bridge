from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np

from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter

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

def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    *, clip: bool = True,
) -> Tuple[QuantumCircuit, List[Parameter]]:
    """Construct a 2‑qubit parameterised circuit that mimics the photonic layers."""
    qc = QuantumCircuit(2)
    param_list: List[Parameter] = []

    def add_layer(params: FraudLayerParameters, clip_flag: bool) -> None:
        # Beam‑splitter analog: apply identical rotations on both qubits
        theta = Parameter(f"theta_{len(param_list)}")
        phi = Parameter(f"phi_{len(param_list)}")
        param_list.extend([theta, phi])
        qc.ry(theta, 0)
        qc.ry(theta, 1)
        qc.rz(phi, 0)
        qc.rz(phi, 1)

        # Phase shifters
        for i, phase in enumerate(params.phases):
            ph = Parameter(f"phase{i}_{len(param_list)}")
            param_list.append(ph)
            qc.rz(ph, i)

        # Squeezing – modelled by phase rotations
        for i, (r, phs) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_p = Parameter(f"squeeze_r{i}_{len(param_list)}")
            ph_p = Parameter(f"squeeze_phi{i}_{len(param_list)}")
            param_list.extend([r_p, ph_p])
            qc.rz(r_p, i)
            qc.rz(ph_p, i)

        # Displacement – modelled by X‑rotations and phase shifts
        for i, (dr, dp) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            dr_p = Parameter(f"disp_r{i}_{len(param_list)}")
            dp_p = Parameter(f"disp_phi{i}_{len(param_list)}")
            param_list.extend([dr_p, dp_p])
            qc.rx(dr_p, i)
            qc.rz(dp_p, i)

        # Kerr non‑linearity – modelled by a phase gate
        for i, k in enumerate(params.kerr):
            k_p = Parameter(f"kerr_{i}_{len(param_list)}")
            param_list.append(k_p)
            qc.rz(k_p, i)

    # Input layer (unclipped)
    add_layer(input_params, clip_flag=False)
    # Subsequent layers (clipped)
    for layer in layers:
        add_layer(layer, clip_flag=True)

    return qc, param_list

class FraudDetectionHybrid:
    """Hybrid quantum fraud‑detection model that evaluates a parameterised
    2‑qubit circuit and returns the expectation of Y on qubit 0."""
    PARAMS_PER_LAYER = 14  # 2 + 2 + 2 + 2 + 2 + 2 + 2

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.circuit, self.params = build_fraud_detection_circuit(input_params, layers)
        self.backend = Aer.get_backend('statevector_simulator')
        self.param_values: dict[Parameter, float] = {}

        def pack_layer(layer: FraudLayerParameters, clip_flag: bool) -> List[float]:
            def clip(v: float, bound: float) -> float:
                return _clip(v, bound) if clip_flag else v
            return [
                clip(layer.bs_theta, 5.0),
                clip(layer.bs_phi, 5.0),
                *[clip(p, 5.0) for p in layer.phases],
                *[clip(r, 5.0) for r in layer.squeeze_r],
                *[clip(p, 5.0) for p in layer.squeeze_phi],
                *[clip(r, 5.0) for r in layer.displacement_r],
                *[clip(p, 5.0) for p in layer.displacement_phi],
                *[clip(k, 1.0) for k in layer.kerr],
            ]

        offset = 0
        # Input layer values
        for param, val in zip(self.params[offset:offset+self.PARAMS_PER_LAYER],
                              pack_layer(input_params, clip_flag=False)):
            self.param_values[param] = val
        offset += self.PARAMS_PER_LAYER

        # Subsequent layers values
        for layer in layers:
            for param, val in zip(self.params[offset:offset+self.PARAMS_PER_LAYER],
                                  pack_layer(layer, clip_flag=True)):
                self.param_values[param] = val
            offset += self.PARAMS_PER_LAYER

    def set_weights(self, weights: dict[Parameter, float]) -> None:
        """Override the internal parameter values."""
        self.param_values.update(weights)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the expectation of Y⊗I on the 2‑qubit state for each input vector.
        `inputs` is an array of shape (batch, 2) where the first two parameters
        of the circuit are treated as the input data and the rest are fixed weights.
        """
        # Bind first two parameters to the input data
        for idx in range(min(2, inputs.shape[1])):
            self.param_values[self.params[idx]] = float(inputs[0, idx])

        bound_qc = self.circuit.bind_parameters(self.param_values)
        result = self.backend.run(bound_qc).result()
        state = result.get_statevector()
        # Expectation of Y on qubit 0: <ψ|Y⊗I|ψ>
        Y = np.array([[0, -1j], [1j, 0]])
        I = np.eye(2)
        op = np.kron(Y, I)
        exp_val = np.real(np.vdot(state, op @ state))
        return np.array([exp_val])

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
