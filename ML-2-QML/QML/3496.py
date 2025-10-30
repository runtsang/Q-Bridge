"""HybridConv: quantum convolution‑pooling network.

This QML module implements a stack of quantum convolution and pooling
circuits inspired by the QCNN.py reference.  Each convolution block
acts on pairs of qubits and is followed by a pooling block that
compresses the entanglement across the circuit.  The depth of the
network is configurable, and the output is a probability of measuring
the computational basis state |1⟩ averaged over all qubits.  The
module is fully compatible with Qiskit Aer simulators and any
Aer backend supporting parameter binding.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Iterable, Optional


class HybridConv:
    """
    Quantum convolution‑pooling network.

    Parameters
    ----------
    kernel_size: int
        Number of qubits per convolutional block (must be even).
    depth: int
        Number of convolution‑pooling layers to stack.
    threshold: float
        Classical threshold used to encode input data into rotation angles.
    backend: Optional[qiskit.providers.Backend]
        Backend for execution; defaults to Aer qasm_simulator.
    shots: int
        Number of shots for circuit execution.
    """

    def __init__(
        self,
        *,
        kernel_size: int = 4,
        depth: int = 3,
        threshold: float = 0.0,
        backend: Optional[qiskit.providers.Backend] = None,
        shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.depth = depth
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Build the variational ansatz
        self.circuit = self._build_ansatz()

        # Estimator and QNN
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=SparsePauliOp.from_list([("Z" + "I" * (kernel_size * depth - 1), 1)]),
            input_params=self._feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Feature mapping: encode classical data as Z rotations
    # ------------------------------------------------------------------
    def _feature_map(self, data: np.ndarray) -> dict:
        """
        Convert an image patch into a dictionary of parameter bindings for
        the feature map.
        """
        flat = data.flatten()
        binds = {}
        for i, val in enumerate(flat):
            # map pixel intensity to a rotation angle
            angle = np.pi if val > self.threshold else 0.0
            binds[self._feature_map.parameters[i]] = angle
        return binds

    # ------------------------------------------------------------------
    # Convolution circuit on a pair of qubits
    # ------------------------------------------------------------------
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    # ------------------------------------------------------------------
    # Pooling circuit on a pair of qubits
    # ------------------------------------------------------------------
    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ------------------------------------------------------------------
    # Build a layer of convs on adjacent pairs
    # ------------------------------------------------------------------
    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(param_prefix, length=num_qubits * 3 // 2)
        i = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(self._conv_circuit(params[i : i + 3]), [q1, q2])
            qc.barrier()
            i += 3
        return qc

    # ------------------------------------------------------------------
    # Build a pooling layer that reduces qubit count by half
    # ------------------------------------------------------------------
    def _pool_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        i = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            qc.append(self._pool_circuit(params[i : i + 3]), [q1, q2])
            qc.barrier()
            i += 3
        return qc

    # ------------------------------------------------------------------
    # Assemble the full ansatz circuit
    # ------------------------------------------------------------------
    def _build_ansatz(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.kernel_size * self.depth)

        # Feature map
        self._feature_map = ZFeatureMap(self.kernel_size * self.depth)
        qc.compose(self._feature_map, range(self.kernel_size * self.depth), inplace=True)

        current_qubits = self.kernel_size * self.depth
        for d in range(self.depth):
            # Convolution
            conv = self._conv_layer(current_qubits, f"c{d}")
            qc.append(conv, range(current_qubits))
            # Pooling (halving qubits)
            pool = self._pool_layer(current_qubits, f"p{d}")
            qc.append(pool, range(current_qubits))
            current_qubits //= 2

        return qc.decompose()

    # ------------------------------------------------------------------
    # Run the QNN on a single input patch
    # ------------------------------------------------------------------
    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the quantum circuit on a 2‑D array of shape
        (kernel_size, kernel_size) or flattened 1‑D array.

        Returns
        -------
        float
            Probability of measuring the computational basis state |1⟩
            averaged over all qubits.
        """
        if data.ndim == 2:
            data = data.reshape(1, -1)
        elif data.ndim == 1:
            data = data.reshape(1, -1)

        binds = [self._feature_map(d) for d in data]
        result = self.qnn.evaluate(np.array([]), binds, self.backend)
        # The EstimatorQNN returns expectation value of the observable
        # which is (1 - 2 * P(|1>)) for Z measurement.  Convert to probability.
        prob = (1 - result[0]) / 2
        return float(prob)

__all__ = ["HybridConv"]
