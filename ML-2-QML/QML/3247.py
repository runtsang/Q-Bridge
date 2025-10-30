import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
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

class HybridFCL:
    """Quantum equivalent of the classical HybridFCL.

    Builds a 2‑qubit parameterised circuit that mimics the photonic fraud‑detection
    layer: entanglement via CX, single‑qubit rotations for phases, squeeze,
    displacement and a Kerr‑like phase gate.  The circuit is executed on a
    QASM simulator and returns the expectation value of the Z observable
    summed over both qubits.
    """

    def __init__(
        self,
        params: FraudLayerParameters,
        backend: qiskit.providers.Backend = None,
        shots: int = 1000,
        clip: bool = True,
    ) -> None:
        self.params = params
        self.clip = clip
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = self._build_circuit(params)

    def _build_circuit(self, params: FraudLayerParameters) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Entanglement block (analogous to a beamsplitter)
        qc.cx(0, 1)
        # Phase rotations
        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)
        # Squeeze analog: rotation around Y
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            angle = _clip(r, 5.0) if self.clip else r
            qc.ry(angle, i)
        # Second entanglement
        qc.cx(0, 1)
        # Displacement analog: rotation around X
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            angle = _clip(r, 5.0) if self.clip else r
            qc.rx(angle, i)
        # Kerr nonlinearity analog: phase gate
        for i, k in enumerate(params.kerr):
            angle = _clip(k, 1.0) if self.clip else k
            qc.p(angle, i)
        qc.measure_all()
        return qc

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit on the backend and return the Z‑observable expectation."""
        theta_param = Parameter("theta")
        bound_dict = {theta_param: thetas[0]}  # single‑parameter demo
        bound_circuit = self.circuit.bind_parameters(bound_dict)
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        # Compute expectation of Z for both qubits
        exp = 0.0
        for bitstring, cnt in counts.items():
            z0 = 1 if bitstring[-1] == '0' else -1  # qubit 0
            z1 = 1 if bitstring[-2] == '0' else -1  # qubit 1
            exp += (z0 + z1) * cnt
        exp /= self.shots
        return np.array([exp])

__all__ = ["FraudLayerParameters", "HybridFCL"]
