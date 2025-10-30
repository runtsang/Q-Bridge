from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

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

def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, clip: bool = False) -> None:
    # Emulate a beamsplitter with simple Rz‑CX‑Rz sequence
    theta = params.bs_theta
    phi = params.bs_phi
    qc.rz(theta, 0)
    qc.cx(0, 1)
    qc.rz(phi, 1)
    # Phase rotations
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)
    # “Squeezing” → Ry with optional clipping
    for i, (r, phi_s) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_clip = r if not clip else _clip(r, 5.0)
        qc.ry(r_clip, i)
    # “Displacement” → Rz with optional clipping
    for i, (r, phi_d) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_clip = r if not clip else _clip(r, 5.0)
        qc.rz(r_clip, i)
    # “Kerr” → tighter clipping on Rz
    for i, k in enumerate(params.kerr):
        k_clip = k if not clip else _clip(k, 1.0)
        qc.rz(k_clip, i)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "build_classifier_circuit"]
