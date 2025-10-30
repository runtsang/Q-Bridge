"""Hybrid quantum sampler network combining convolutional layers and a sampler block.

This module builds a quantum circuit that merges the QCNN ansatz (convolution and pooling)
with a SamplerQNN‑style two‑qubit block. The resulting EstimatorQNN is wrapped in a
class :class:`HybridSamplerQNN` that exposes a :meth:`forward` method compatible with
PyTorch‑style APIs. The circuit is parameterized by input features (through a
ZFeatureMap) and trainable weights for both the ansatz and the sampler.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


def _build_qnn() -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Convolution block
    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = conv_circuit(params[idx : idx + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            idx += 3
        for q1, q2 in zip(range(1, num_qubits, 2), list(range(0, num_qubits, 2)) + [0]):
            sub = conv_circuit(params[idx : idx + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    # Pooling block
    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            sub = pool_circuit(params[idx : idx + 3])
            qc.append(sub, [src, sink])
            qc.barrier()
            idx += 3
        return qc

    # Ansätze
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Sampler block (two‑qubit Ry + CX)
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    sampler_circ = QuantumCircuit(2)
    sampler_circ.ry(inputs[0], 0)
    sampler_circ.ry(inputs[1], 1)
    sampler_circ.cx(0, 1)
    sampler_circ.ry(weights[0], 0)
    sampler_circ.ry(weights[1], 1)
    sampler_circ.cx(0, 1)
    sampler_circ.ry(weights[2], 0)
    sampler_circ.ry(weights[3], 1)

    # Assemble full circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    circuit.append(sampler_circ, [0, 1])

    # Observable on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Build EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=list(ansatz.parameters) + list(sampler_circ.parameters),
        estimator=estimator,
    )
    return qnn


class HybridSamplerQNN:
    """Quantum wrapper exposing a PyTorch‑style :meth:`forward` interface.

    The underlying EstimatorQNN performs the forward pass. The class is
    intentionally lightweight to keep the quantum backend isolated.
    """

    def __init__(self) -> None:
        self._qnn = _build_qnn()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run a forward pass.

        Parameters
        ----------
        inputs
            Input feature array of shape (batch, 8).

        Returns
        -------
        np.ndarray
            Expectation values for each input sample.
        """
        return self._qnn.predict(inputs)

    def parameters(self):
        """Return the trainable parameters of the underlying EstimatorQNN."""
        return self._qnn.weight_params


__all__ = ["HybridSamplerQNN"]
