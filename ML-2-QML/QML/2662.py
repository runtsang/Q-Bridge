"""Quantum QCNN implementation with depth‑controlled convolution and pooling.

The quantum network is constructed from reusable conv_layer and pool_layer
sub‑circuits.  Each layer is parameterised by a ParameterVector; the total
number of parameters grows linearly with *depth*.  The feature map is a
ZFeatureMap that embeds classical data into the quantum state.  The final
EstimatorQNN wraps the circuit for training with a classical optimiser.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class QCNNGen226:
    """Depth‑controlled quantum convolutional neural network.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (must match the feature‑map dimension).
    depth : int
        Number of convolution/pooling pairs.  Each pair consumes 3 parameters
        per qubit, so the total parameter count is ``3 * num_qubits * depth``.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, num_qubits: int = 8, depth: int = 3, seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        if seed is not None:
            algorithm_globals.random_seed = seed

        # Feature map embedding classical data.
        self.feature_map = ZFeatureMap(num_qubits)

        # Build the ansatz circuit.
        self.ansatz = QuantumCircuit(num_qubits, name="Ansatz")
        self._build_ansatz()

        # Prepare the EstimatorQNN.
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=[SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])],
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Helper sub‑circuits
    # ------------------------------------------------------------------
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single two‑qubit convolution unit."""
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

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling unit (no measurement)."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, qubits: List[int], prefix: str) -> QuantumCircuit:
        """Compose convolution blocks across the given qubit pairs."""
        params = ParameterVector(prefix, length=len(qubits) // 2 * 3)
        qc = QuantumCircuit(self.num_qubits)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[idx: idx + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(self, sources: List[int], sinks: List[int], prefix: str) -> QuantumCircuit:
        """Compose pooling blocks mapping sources to sinks."""
        params = ParameterVector(prefix, length=len(sources) * 3)
        qc = QuantumCircuit(self.num_qubits)
        idx = 0
        for src, snk in zip(sources, sinks):
            sub = self._pool_circuit(params[idx: idx + 3])
            qc.append(sub, [src, snk])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> None:
        """Assemble the full QCNN ansatz with the requested depth."""
        # First layer operates on all qubits
        self.ansatz.compose(self._conv_layer(list(range(self.num_qubits)), "c1"), inplace=True)

        # First pooling reduces the qubit count by half
        half = self.num_qubits // 2
        self.ansatz.compose(
            self._pool_layer(list(range(half)), list(range(half, self.num_qubits)), "p1"),
            inplace=True,
        )

        # Subsequent layers operate on the reduced qubit register
        for d in range(2, self.depth + 1):
            curr = half // (2 ** (d - 2))
            self.ansatz.compose(
                self._conv_layer(list(range(curr)), f"c{d}"),
                inplace=True,
            )
            if curr > 1:
                sink = [i + curr for i in range(curr // 2)]
                self.ansatz.compose(
                    self._pool_layer(list(range(curr // 2)), sink, f"p{d}"),
                    inplace=True,
                )

    def get_qnn(self) -> EstimatorQNN:
        """Return the fully‑constructed EstimatorQNN."""
        return self.qnn

def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple variational classifier circuit.

    The circuit follows the pattern from the reference: an explicit feature
    encoding via ``rx`` gates followed by a depth‑controlled variational
    block consisting of ``ry`` rotations and nearest‑neighbour ``cz`` gates.
    Observables are single‑qubit Z operators, one per qubit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Number of variational layers.

    Returns
    -------
    QuantumCircuit
        The full circuit ready for use with EstimatorQNN or a classical optimiser.
    Iterable
        ParameterVector representing the input encoding.
    Iterable
        ParameterVector representing the variational weights.
    List[SparsePauliOp]
        List of Z observables, one per qubit.
    """
    encoding = ParameterVector("x", length=num_qubits)
    weights = ParameterVector("theta", length=num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for qubit, param in zip(range(num_qubits), encoding):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp.from_list([("I" * i + "Z" + "I" * (num_qubits - i - 1), 1)])
        for i in range(num_qubits)
    ]
    return qc, encoding, weights, observables

__all__ = ["QCNNGen226", "build_classifier_circuit"]
