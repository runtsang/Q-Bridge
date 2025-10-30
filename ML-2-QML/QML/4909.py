import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, execute
from qiskit import Aer
from typing import Iterable


@dataclass
class FraudLayerParameters:
    """Parameters describing a layer in the quantum model."""
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


class QuantumSelfAttention:
    """Basic quantum circuit representing a self‑attention style block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[i], i)
            qc.ry(entangle_params[i], i)
        for i in range(self.n_qubits - 1):
            qc.crx(0.5, i, i + 1)
        qc.measure_all()
        return qc


def _apply_layer(circuit: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    # Rotation gates
    circuit.ry(_clip(params.bs_theta, 5.0) if clip else params.bs_theta, 0)
    circuit.ry(_clip(params.bs_phi, 5.0) if clip else params.bs_phi, 1)

    # Phase gates
    for i, phase in enumerate(params.phases):
        circuit.rz(_clip(phase, 5.0) if clip else phase, i)

    # Squeeze (interpreted as RX)
    for i, r in enumerate(params.squeeze_r):
        circuit.rx(_clip(r, 5.0) if clip else r, i)

    # Displacement (interpreted as RY)
    for i, r in enumerate(params.displacement_r):
        circuit.ry(_clip(r, 5.0) if clip else r, i)

    # Kerr (interpreted as RZ)
    for i, k in enumerate(params.kerr):
        circuit.rz(_clip(k, 1.0) if clip else k, i)

    # Self‑attention sub‑circuit
    sa = QuantumSelfAttention(2)
    sa_circ = sa._build_circuit(
        np.array(params.squeeze_r), np.array(params.displacement_r)
    )
    circuit.append(sa_circ.to_gate(), [0, 1])


def build_fraud_detection_qiskit_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a Qiskit circuit for the hybrid fraud‑detection model."""
    circuit = QuantumCircuit(2)
    _apply_layer(circuit, input_params, clip=False)
    for layer in layers:
        _apply_layer(circuit, layer, clip=True)
    return circuit


def expectation_z(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    exp = sum(((-1) ** int(state[0])) * count for state, count in counts.items())
    return exp / total


class FraudDetectionQuantumCircuit:
    """Wrapper that executes the fraud‑detection circuit on a backend and returns the Z‑expectation."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = build_fraud_detection_qiskit_program(input_params, layers)

    def run(self) -> float:
        compiled = transpile(self.circuit, self.backend)
        job = execute(compiled, self.backend, shots=self.shots)
        result = job.result().get_counts()
        return expectation_z(result)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_qiskit_program",
    "FraudDetectionQuantumCircuit",
    "QuantumSelfAttention",
]
