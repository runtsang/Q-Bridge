"""Quantum implementation of the QCNN architecture.

The quantum circuit reproduces the classical layer structure:
convolutional and pooling sub‑circuits built from the `conv_circuit`
and `pool_circuit` primitives.  After three rounds of conv/pool the
output qubits are fed into a parameterised fully‑connected layer
(`FCLCircuit`) and finally evaluated with a `EstimatorQNN` wrapper.
The resulting expectation value is the network output.

The class name `QCNNHybrid` matches the classical counterpart so
that both modules expose a uniform API.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Iterable


# ----------------------------------------------------------------------
# Convolution and pooling primitives (from reference 1)
# ----------------------------------------------------------------------
def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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


def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
        qc.compose(conv_circuit(params[i * 3 : i * 3 + 3]), [q1, q2], inplace=True)
        qc.barrier()
    return qc


def pool_circuit(params: ParameterVector) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def pool_layer(sources: Iterable[int], sinks: Iterable[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for i, (src, snk) in enumerate(zip(sources, sinks)):
        qc.compose(pool_circuit(params[i * 3 : i * 3 + 3]), [src, snk], inplace=True)
        qc.barrier()
    return qc


# ----------------------------------------------------------------------
# Fully‑connected layer (from reference 2)
# ----------------------------------------------------------------------
class FCLCircuit(QuantumCircuit):
    """Parameterised single‑qubit circuit that mimics a fully‑connected layer."""

    def __init__(self, n_qubits: int = 1, backend=Aer.get_backend("qasm_simulator"), shots: int = 1024):
        super().__init__(n_qubits)
        self.theta = Parameter("theta")
        self.h(range(n_qubits))
        self.barrier()
        self.ry(self.theta, range(n_qubits))
        self.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(
            self,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        result = job.result()
        counts = result.get_counts(self)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(state, 2) for state in counts.keys()], dtype=float)
        return np.sum(states * probs)


# ----------------------------------------------------------------------
# Hybrid QCNN quantum network
# ----------------------------------------------------------------------
class QCNNHybrid:
    """
    Quantum version of the hybrid QCNN.

    The circuit is built in three conv–pool stages followed by a
    parameterised fully‑connected circuit (`FCLCircuit`).  The final
    expectation value is computed with an `EstimatorQNN` wrapper.
    """

    def __init__(self) -> None:
        # Feature map (ZFeatureMap) – omitted for brevity; use simple identity
        self.feature_map = QuantumCircuit(8)
        self.feature_map.barrier()

        # Build the ansatz
        self.ansatz = QuantumCircuit(8, name="Ansatz")
        self.ansatz.compose(conv_layer(8, "c1"), inplace=True)
        self.ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
        self.ansatz.compose(conv_layer(4, "c2"), inplace=True)
        self.ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
        self.ansatz.compose(conv_layer(2, "c3"), inplace=True)
        self.ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

        # Fully‑connected layer
        self.fcl = FCLCircuit(1)

        # Combine feature map, ansatz and FC circuit
        self.circuit = QuantumCircuit(8)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)

        # Observable for expectation value
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

        # EstimatorQNN wrapper
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Return the network output for the given classical inputs."""
        return self.qnn.predict(inputs)


def QCNNHybrid() -> QCNNHybrid:
    """Factory returning the quantum QCNNHybrid."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "QCNNHybrid"]
