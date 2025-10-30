from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
from typing import Iterable, Tuple


def _clip(value: float, bound: float) -> float:
    """Clamp a real value to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


class ConvGen396:
    """
    Quantum convolution (quanvolution) filter.  It maps a 2‑D kernel to a
    single expectation value by encoding each pixel into a RX rotation
    and running a shallow random circuit for entanglement.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 100,
        threshold: float = 0.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits)
        for i, p in enumerate(self.theta):
            qc.rx(p, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """Execute the quanvolution on a kernel of data."""
        flat = np.reshape(data, (self.n_qubits,))
        param_binds: list[dict[qiskit.circuit.Parameter, float]] = []

        for val in flat:
            bind = {p: np.pi if val > self.threshold else 0.0 for p in self.theta}
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        total_ones = 0
        for bitstring, count in result.items():
            total_ones += count * sum(int(bit) for bit in bitstring)

        return total_ones / (self.shots * self.n_qubits)


def Conv() -> ConvGen396:
    """Factory returning a quantum filter instance."""
    return ConvGen396(
        kernel_size=2,
        backend=qiskit.Aer.get_backend("qasm_simulator"),
        shots=100,
        threshold=127.0,
    )


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[
    QuantumCircuit,
    Iterable[qiskit.circuit.Parameter],
    Iterable[qiskit.circuit.Parameter],
    list[SparsePauliOp],
]:
    """
    Construct a shallow variational ansatz with explicit data‑encoding
    and a set of observables that can be used as a hybrid classifier head.
    Mirrors the quantum helper in the original reference pair.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for p, q in zip(encoding, range(num_qubits)):
        qc.rx(p, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return qc, list(encoding), list(weights), observables
