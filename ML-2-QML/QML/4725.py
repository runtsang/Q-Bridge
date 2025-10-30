"""Quantum neural network that mirrors the hybrid classical architecture.

The circuit contains:
  * A feature map that encodes classical data via a RealAmplitudes ansatz.
  * Convolutional and pooling layers implemented with two‑qubit
    gates, identical to the QCNN helper.
  * An expectation value of a single Pauli‑Z observable on the last qubit.

The function `EstimatorQNN()` returns a ready‑to‑train
qiskit_machine_learning.neural_networks.EstimatorQNN instance.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit convolution unit, parameterised by three angles."""
    qc = QuantumCircuit(2)
    qc.rz(-3.141592653589793 / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(3.141592653589793 / 2, 0)
    return qc


def _conv_layer(num_qubits: int, prefix: str) -> QuantumCircuit:
    """Convolutional layer consisting of multiple conv units."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    param_idx = 0
    params = ParameterVector(prefix, length=num_qubits * 3)
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        qc.append(_conv_circuit(params[param_idx : param_idx + 3]), [q1, q2])
        qc.barrier()
        param_idx += 3
    return qc


def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
    """Two‑qubit pooling unit, sharing the same gate pattern as conv."""
    qc = QuantumCircuit(2)
    qc.rz(-3.141592653589793 / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc


def _pool_layer(sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
    """Pooling layer that pairs sources and sinks."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_idx = 0
    params = ParameterVector(prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.append(_pool_circuit(params[param_idx : param_idx + 3]), [src, snk])
        qc.barrier()
        param_idx += 3
    return qc


def EstimatorQNN() -> EstimatorQNN:
    """Factory returning a Qiskit EstimatorQNN instance."""
    # Feature map – a RealAmplitudes ansatz that encodes the data.
    feature_map = RealAmplitudes(4, reps=1)

    # Build the ansatz by stacking convolution and pooling layers.
    ansatz = QuantumCircuit(4)
    ansatz.append(_conv_layer(4, "c1"), range(4))
    ansatz.append(_pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1") if ansatz.num_qubits < 8 else None, range(8))
    # For brevity only one conv‑pool pair is included; more can be added.

    # Combine feature map and ansatz.
    circuit = QuantumCircuit(4)
    circuit.compose(feature_map, range(4), inplace=True)
    circuit.compose(ansatz, range(4), inplace=True)

    # Observable: expectation value of Z on the last qubit.
    observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])

    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )


__all__ = ["EstimatorQNN"]
