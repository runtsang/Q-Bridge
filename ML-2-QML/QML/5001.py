import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

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

def build_fraud_detection_program_qiskit(input_params: FraudLayerParameters,
                                        layers: Iterable[FraudLayerParameters]) -> QuantumCircuit:
    """Construct a Qiskit circuit mirroring the photonic fraud‑detection program."""
    qc = QuantumCircuit(2)
    def _apply_layer(qc: QuantumCircuit,
                     params: FraudLayerParameters,
                     clip: bool) -> None:
        qc.append(qiskit.circuit.library.BSgate(theta=params.bs_theta,
                                                phi=params.bs_phi), [0, 1])
        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qc.s(r if not clip else _clip(r, 5), i)
        qc.append(qiskit.circuit.library.BSgate(theta=params.bs_theta,
                                                phi=params.bs_phi), [0, 1])
        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)
        for i, (r, phi) in enumerate(zip(params.displacement_r,
                                         params.displacement_phi)):
            qc.p(r if not clip else _clip(r, 5), i)
        for i, k in enumerate(params.kerr):
            qc.rz(k if not clip else _clip(k, 1), i)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc

def FCL_qiskit(n_qubits: int = 1, shots: int = 1024) -> Tuple[QuantumCircuit, Parameter]:
    """Create a parameterised quantum circuit resembling a fully‑connected layer."""
    qc = QuantumCircuit(n_qubits)
    theta = Parameter("theta")
    qc.h(range(n_qubits))
    qc.ry(theta, range(n_qubits))
    qc.measure_all()
    return qc, theta

class HybridFraudQLSTM:
    """
    Quantum hybrid model that chains a fraud‑detection circuit,
    an optional quantum LSTM block, and a fully‑connected layer.
    """
    def __init__(self,
                 fraud_params: Iterable[FraudLayerParameters],
                 n_qubits: int = 0,
                 shots: int = 1024) -> None:
        self.circuit = build_fraud_detection_program_qiskit(fraud_params[0], fraud_params[1:])
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        if n_qubits > 0:
            self._add_quantum_lstm()

    def _add_quantum_lstm(self) -> None:
        qc_lstm = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc_lstm.rx(Parameter(f"phi_{i}"), i)
        for i in range(self.n_qubits - 1):
            qc_lstm.cx(i, i + 1)
        self.circuit.append(qc_lstm, range(self.n_qubits))

    def add_fcl_layer(self, n_qubits: int = 1, shots: int = 1024) -> None:
        fcl, theta = FCL_qiskit(n_qubits, shots)
        self.circuit.append(fcl, range(n_qubits))
        self.circuit.parameters.update({theta: 0.0})

    def run(self,
            parameter_sets: Sequence[Sequence[float]]) -> np.ndarray:
        results = []
        for params in parameter_sets:
            job = execute(self.circuit,
                          self.backend,
                          shots=self.shots,
                          parameter_binds=[{p: val for p, val in zip(self.circuit.parameters, params)}])
            counts = job.result().get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys())).astype(float)
            expectation = np.sum(states * probs)
            results.append([expectation])
        return np.array(results)

    def evaluate_with_shots(self,
                            parameter_sets: Sequence[Sequence[float]],
                            shots: int) -> np.ndarray:
        self.shots = shots
        return self.run(parameter_sets)

__all__ = ["HybridFraudQLSTM", "FraudLayerParameters", "build_fraud_detection_program_qiskit", "FCL_qiskit"]
