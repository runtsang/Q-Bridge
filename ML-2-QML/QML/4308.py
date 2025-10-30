"""Quantum hybrid architecture combining QCNN ansatz and photonic fraud‑detection circuit.

The class provides:
* `qcnn_qnn()` – returns a Qiskit EstimatorQNN implementing the QCNN variational circuit.
* `photonic_program()` – returns a Strawberry Fields program for the fraud‑detection layer.
* `evaluate_qcnn(qnn, features)` – runs the QCNN QNN on a feature vector.
* `evaluate_photonic(program, inputs)` – evaluates the photonic circuit on an input vector.

Usage:
    model = UnifiedQCNNFraudQuanvolution()
    qnn = model.qcnn_qnn()
    qcnn_result = model.evaluate_qcnn(qnn, feature_vector)
    program = model.photonic_program()
    photonic_result = model.evaluate_photonic(program, input_vector)
"""

from __future__ import annotations

import numpy as np

# Qiskit imports for QCNN ansatz
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Strawberry Fields imports for photonic circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

class UnifiedQCNNFraudQuanvolution:
    """Quantum hybrid architecture combining QCNN ansatz and photonic fraud‑detection circuit."""

    def __init__(self) -> None:
        # Seed random generator for reproducibility
        np.random.seed(42)

    # ------------------------------------------------------------------
    # QCNN variational circuit
    # ------------------------------------------------------------------
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit convolution block used in the QCNN ansatz."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Construct a convolutional layer over `num_qubits` qubits."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        idx = 0
        for q in range(0, num_qubits - 1, 2):
            block = self._conv_circuit(params[idx:idx + 3])
            qc.append(block, [q, q + 1])
            idx += 3
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling block used in the QCNN ansatz."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Construct a pooling layer over `num_qubits` qubits."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3 // 2)
        idx = 0
        for q in range(0, num_qubits - 1, 2):
            block = self._pool_circuit(params[idx:idx + 3])
            qc.append(block, [q, q + 1])
            idx += 3
        return qc

    def qcnn_qnn(self) -> EstimatorQNN:
        """Return a Qiskit EstimatorQNN object implementing the QCNN ansatz."""
        num_qubits = 8
        qc = QuantumCircuit(num_qubits)

        # Feature map (Z‑feature map) – simple Pauli‑Z rotations
        for i in range(num_qubits):
            qc.ry(np.random.rand() * 2 * np.pi, i)

        # Build QCNN ansatz
        qc.append(self._conv_layer(num_qubits, "c1"), range(num_qubits))
        qc.append(self._pool_layer(num_qubits, "p1"), range(num_qubits))

        # Down‑sample to 4 qubits
        qc.barrier()
        for i in range(0, num_qubits, 2):
            qc.cx(i, i + 1)
            qc.measure(i, i)  # measurement for classical read‑out

        # Second stage
        qc.append(self._conv_layer(4, "c2"), range(4))
        qc.append(self._pool_layer(4, "p2"), range(4))

        # Final measurement
        observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])
        estimator = Estimator()
        qnn = EstimatorQNN(
            circuit=qc.decompose(),
            observables=observable,
            input_params=[],  # no explicit feature parameters
            weight_params=qc.parameters,
            estimator=estimator,
        )
        return qnn

    def evaluate_qcnn(self, qnn: EstimatorQNN, features: np.ndarray) -> np.ndarray:
        """Run the QCNN QNN on a batch of feature vectors."""
        return qnn.predict(features)

    # ------------------------------------------------------------------
    # Photonic fraud‑detection circuit
    # ------------------------------------------------------------------
    def photonic_program(self) -> sf.Program:
        """Return a Strawberry Fields program mimicking the fraud‑detection photonic layer."""
        program = sf.Program(2)
        with program.context as q:
            # Example parameters – in practice these would be trained
            BSgate(0.5, 0.1) | (q[0], q[1])
            Rgate(0.3) | q[0]
            Rgate(0.4) | q[1]
            Sgate(0.2, 0.0) | q[0]
            Sgate(0.3, 0.1) | q[1]
            Dgate(0.5, 0.0) | q[0]
            Dgate(0.6, 0.1) | q[1]
            Kgate(0.1) | q[0]
            Kgate(0.2) | q[1]
        return program

    def evaluate_photonic(self, program: sf.Program, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the photonic program on a batch of input vectors."""
        from strawberryfields import Engine
        eng = Engine("gaussian")
        results = []
        for inp in inputs:
            eng.run(program, args={"q[0]": inp[0], "q[1]": inp[1]})
            state = eng.run(program).state
            # Expectation of photon number as a simple observable
            photon_counts = state.expectation_value(sf.ops.PauliZ(), [0, 1])
            results.append(photon_counts)
        return np.array(results)

__all__ = ["UnifiedQCNNFraudQuanvolution"]
