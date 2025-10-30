import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class FraudDetectionHybrid:
    """Quantum fraud‑detection circuit built with Qiskit."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        self.input_params = input_params
        self.layers = list(layers)
        self.qreg = QuantumRegister(2, name="q")
        self.creg = ClassicalRegister(1, name="c")
        self.circuit = QuantumCircuit(self.qreg, self.creg)
        self._build_circuit()

    def _apply_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        # Simulated beam‑splitter using a rotation‑plus‑CNOT construction
        theta = params.bs_theta
        phi = params.bs_phi
        self.circuit.ry(theta, self.qreg[0])
        self.circuit.ry(theta, self.qreg[1])
        self.circuit.cx(self.qreg[0], self.qreg[1])
        self.circuit.ry(phi, self.qreg[0])
        self.circuit.ry(phi, self.qreg[1])

        # Phase shifts
        for i, phase in enumerate(params.phases):
            self.circuit.rz(phase, self.qreg[i])

        # Approximate squeezing and displacement with single‑qubit rotations
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            self.circuit.rx(_clip(r, 5.0), self.qreg[i])
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            self.circuit.rz(_clip(r, 5.0), self.qreg[i])

        # Kerr nonlinearity approximated with an RZ rotation
        for i, k in enumerate(params.kerr):
            self.circuit.rz(_clip(k, 1.0), self.qreg[i])

    def _build_circuit(self):
        self._apply_layer(self.input_params, clip=False)
        for layer in self.layers:
            self._apply_layer(layer, clip=True)
        self.circuit.measure(self.qreg[0], self.creg[0])

    def evaluate(self, shots: int = 1024) -> float:
        backend = Aer.get_backend("qasm_simulator")
        job = execute(self.circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        # return the probability of measuring the qubit in state |1⟩
        return counts.get('1', 0) / shots

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
