"""Combined quantum convolution and classifier module.

The ConvGen212 class encapsulates a quantum convolution filter and a variational classifier circuit.
It mirrors the classical counterpart while leveraging Qiskit for quantum simulation.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List


class ConvGen212:
    """Quantum convolution + classifier hybrid.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel (defines number of qubits).
    shots : int
        Number of shots for the simulator.
    threshold : float
        Threshold for encoding data into parameterized rotations.
    num_qubits : int
        Number of qubits for the classifier circuit.
    depth : int
        Depth of the classifier ansatz.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 100,
        threshold: float = 127,
        num_qubits: int = 4,
        depth: int = 2,
    ) -> None:
        # Quantum convolution
        self.kernel_size = kernel_size
        self.n_qubits_conv = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.conv_circuit = self._build_conv_circuit()

        # Quantum classifier
        self.num_qubits_cls = num_qubits
        self.depth_cls = depth
        (
            self.cls_circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = self._build_classifier_circuit()

    def _build_conv_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits_conv)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits_conv)]
        for i in range(self.n_qubits_conv):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(self.n_qubits_conv, depth=2)
        qc.measure_all()
        return qc

    def run_conv(self, data: np.ndarray) -> float:
        """Execute the quantum convolution filter on classical data."""
        data = np.reshape(data, (1, self.n_qubits_conv))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.conv_circuit.parameters[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self.conv_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.conv_circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits_conv)

    def _build_classifier_circuit(
        self,
    ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits_cls)
        weights = ParameterVector("theta", self.num_qubits_cls * self.depth_cls)

        qc = QuantumCircuit(self.num_qubits_cls)
        for param, qubit in zip(encoding, range(self.num_qubits_cls)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth_cls):
            for qubit in range(self.num_qubits_cls):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits_cls - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits_cls - i - 1))
            for i in range(self.num_qubits_cls)
        ]
        return qc, list(encoding), list(weights), observables

    def run_classifier(self, data: np.ndarray) -> List[float]:
        """Execute the quantum classifier circuit on classical data."""
        # Encode data into the circuit parameters
        param_binds = {self.encoding[i]: float(val) for i, val in enumerate(data[: self.num_qubits_cls])}
        # Set variational parameters to zero for a deterministic run
        for w in self.weights:
            param_binds[w] = 0.0

        job = execute(
            self.cls_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        result = job.result().get_counts(self.cls_circuit)

        # Compute expectation values for each Z observable
        expectations: List[float] = []
        for i in range(self.num_qubits_cls):
            ones = 0
            for key, val in result.items():
                if key[self.num_qubits_cls - 1 - i] == "1":
                    ones += val
            prob1 = ones / self.shots
            expectations.append(1 - 2 * prob1)
        return expectations

    def classify(self, data: np.ndarray) -> List[float]:
        """Run convolution followed by classifier."""
        conv_out = self.run_conv(data)
        # Pad the scalar output to match the classifier input size
        feature = np.array([conv_out] * self.num_qubits_cls)
        return self.run_classifier(feature)

    def run(self, data) -> float:
        """Compatibility wrapper that accepts numpy arrays for convolution."""
        return self.run_conv(data)


def Conv() -> ConvGen212:
    """Return a ConvGen212 instance as a drop‑in replacement."""
    return ConvGen212()


__all__ = ["ConvGen212", "Conv"]
