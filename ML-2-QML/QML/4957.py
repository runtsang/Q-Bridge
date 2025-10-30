from __future__ import annotations

from typing import Iterable, List, Any
import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class HybridFullyConnectedLayer:
    """
    Quantum equivalent of the classical hybrid layer.
    Builds a parameterized circuit with encoding, depth, optional attention
    entanglement, and fraud‑style angle clipping.
    """
    def __init__(
        self,
        num_qubits: int = 1,
        depth: int = 1,
        use_attention: bool = False,
        use_fraud_clip: bool = False,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_attention = use_attention
        self.use_fraud_clip = use_fraud_clip

        # Parameter vectors
        self.encoding = ParameterVector("x", num_qubits)
        self.theta = ParameterVector("theta", num_qubits * depth)
        if use_attention:
            self.attention_theta = ParameterVector(
                "attn", (num_qubits - 1) * depth
            )

        self.circuit = QuantumCircuit(num_qubits)

        # Encoding
        for i in range(num_qubits):
            self.circuit.rx(self.encoding[i], i)

        # Ansatz
        for d in range(depth):
            for q in range(num_qubits):
                self.circuit.ry(self.theta[d * num_qubits + q], q)
            if use_attention:
                for q in range(num_qubits - 1):
                    idx = d * (num_qubits - 1) + q
                    self.circuit.crx(self.attention_theta[idx], q, q + 1)
            else:
                for q in range(num_qubits - 1):
                    self.circuit.cz(q, q + 1)

        self.circuit.measure_all()

        # Observables for expectation
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

    def run(
        self,
        backend: Any,
        thetas: Iterable[float],
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit with the given parameters.
        ``thetas`` should match the total number of variational parameters.
        Returns expectation values for the Z observables.
        """
        param_dict: dict[str, float] = {}
        idx = 0
        for p in self.theta:
            value = float(thetas[idx])
            if self.use_fraud_clip:
                value = _clip(value, np.pi)  # clip to [-π, π]
            param_dict[str(p)] = value
            idx += 1

        if self.use_attention:
            for p in self.attention_theta:
                value = float(thetas[idx])
                if self.use_fraud_clip:
                    value = _clip(value, np.pi)
                param_dict[str(p)] = value
                idx += 1

        job = execute(
            self.circuit,
            backend,
            shots=shots,
            parameter_binds=[param_dict],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        expectations: List[float] = []
        for qubit in range(self.num_qubits):
            exp = 0.0
            for state, cnt in counts.items():
                # Qiskit returns bitstrings with qubit 0 as the most significant bit
                bit = int(state[::-1][qubit])  # reverse to match qubit order
                exp += (1 if bit else -1) * cnt
            exp /= shots
            expectations.append(exp)

        return np.array(expectations)

def FCL() -> HybridFullyConnectedLayer:
    """
    Factory returning a hybrid quantum fully‑connected layer.
    Uses depth=2, attention on, fraud‑style clipping enabled.
    """
    return HybridFullyConnectedLayer(
        num_qubits=2,
        depth=2,
        use_attention=True,
        use_fraud_clip=True,
    )
