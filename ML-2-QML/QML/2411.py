import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from dataclasses import dataclass
from typing import Iterable, List

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

class FraudDetectionHybridQuantum:
    """Hybrid photonic‑plus‑quantum fraud‑detection program."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: List[FraudLayerParameters]):
        self.input_params = input_params
        self.layers = layers
        self.photonic_prog = self._build_photonic()
        self.attention_circuit = self._build_attention_circuit()

    def _build_photonic(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)
        return prog

    def _apply_layer(self, modes: List[object], params: FraudLayerParameters, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else _clip(k, 1)) | modes[i]

    def _build_attention_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        qc = QuantumCircuit(qr, cr)
        # Example parameter generation – in practice these would be learnable or data‑driven
        rotation_params = np.random.rand(12)
        entangle_params = np.random.rand(3)
        for i in range(4):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(3):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(qr, cr)
        return qc

    def run(self,
            photonic_backend: sf.backends.Backend | None = None,
            attention_backend: qiskit.providers.Backend | None = None):
        """Execute both photonic and quantum self‑attention sub‑circuits."""
        if photonic_backend is None:
            photonic_backend = sf.backends.Simulator()
        photonic_state = photonic_backend.run(self.photonic_prog).state
        if attention_backend is None:
            attention_backend = qiskit.Aer.get_backend("qasm_simulator")
        attention_result = qiskit.execute(self.attention_circuit,
                                          attention_backend,
                                          shots=1024).result()
        return photonic_state, attention_result.get_counts(self.attention_circuit)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> FraudDetectionHybridQuantum:
    """Create a FraudDetectionHybridQuantum instance ready for simulation."""
    return FraudDetectionHybridQuantum(input_params, list(layers))

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybridQuantum",
]
