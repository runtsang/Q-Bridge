from dataclasses import dataclass
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

@dataclass
class HybridFraudParameters:
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

class HybridFraudClassifier:
    """Quantum circuit builder for fraud detection that mirrors the classical architecture.

    The circuit uses a simple layered ansatz: each layer applies an RX encoding,
    followed by a sequence of RY rotations (variational weights) and CZ entangling
    gates. All parameters are clipped to match the bounds used in the classical
    model. Observables are Pauli‑Z on each qubit.
    """

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Encoding stage
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Observables: Pauli‑Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables

    @staticmethod
    def clip_parameters(params: Iterable[HybridFraudParameters], bound: float = 5.0) -> List[HybridFraudParameters]:
        """Clip all numeric fields of the parameters to the given bound."""
        clipped = []
        for p in params:
            clipped.append(
                HybridFraudParameters(
                    bs_theta=_clip(p.bs_theta, bound),
                    bs_phi=_clip(p.bs_phi, bound),
                    phases=tuple(_clip(v, bound) for v in p.phases),
                    squeeze_r=tuple(_clip(v, bound) for v in p.squeeze_r),
                    squeeze_phi=tuple(_clip(v, bound) for v in p.squeeze_phi),
                    displacement_r=tuple(_clip(v, bound) for v in p.displacement_r),
                    displacement_phi=tuple(_clip(v, bound) for v in p.displacement_phi),
                    kerr=tuple(_clip(v, 1.0) for v in p.kerr),
                )
            )
        return clipped

__all__ = ["HybridFraudParameters", "HybridFraudClassifier"]
