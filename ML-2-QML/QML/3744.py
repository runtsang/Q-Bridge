"""Quantum convolutional network that emulates the classical Conv filter while
augmenting it with QCNN‑style convolution and pooling layers.

The implementation uses Qiskit’s EstimatorQNN to provide a differentiable
quantum circuit.  The network accepts a 2‑D array, encodes it as a feature
map, applies a series of parameterised convolution and pooling layers,
and returns the expectation value of a Pauli‑Z observable.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as EstimatorPrimitive
from qiskit.circuit import ParameterVector

class Conv:
    """
    Quantum hybrid convolutional network.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the input image grid.  The image is flattened to
        :math:`kernel_size^2` qubits.
    shots : int, default 100
        Number of shots used by the backend.
    threshold : float, default 127.0
        Threshold for binary encoding of input data.
    backend : qiskit.providers.Backend, optional
        Qiskit backend; defaults to Aer’s qasm simulator.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 100,
                 threshold: float = 127.0,
                 backend: qiskit.providers.Backend = None) -> None:
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2
        self._build_qnn()

    def _build_qnn(self):
        # Feature map
        self.feature_map = ZFeatureMap(self.n_qubits, reps=1, entanglement='linear')
        n_qubits = self.n_qubits

        # Ansatz: a single convolution–pool pair
        ansatz = QuantumCircuit(n_qubits, name="Ansatz")

        # Convolution layer over pairs (0,1) and (2,3)
        ansatz.compose(self._conv_layer(n_qubits, "c1"), range(n_qubits), inplace=True)
        # Pooling layer over pairs (0,1) and (2,3)
        ansatz.compose(self._pool_layer(n_qubits, "p1"), range(n_qubits), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(n_qubits)
        circuit.compose(self.feature_map, range(n_qubits), inplace=True)
        circuit.compose(ansatz, range(n_qubits), inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])

        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=EstimatorPrimitive(backend=self.backend)
        )

    def _conv_layer(self, n_qubits: int, prefix: str) -> QuantumCircuit:
        """Convolution layer composed of a fixed 2‑qubit circuit."""
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector(prefix, length=n_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, n_qubits, 2), range(1, n_qubits, 2)):
            sub = self._conv_circuit(params[idx:idx+3], q1, q2)
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def _conv_circuit(self, params, q1, q2):
        """Two‑qubit convolution unitary."""
        qc = QuantumCircuit(2)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.rx(params[2], 1)
        return qc

    def _pool_layer(self, n_qubits: int, prefix: str) -> QuantumCircuit:
        """Pooling layer over pairs (0,1), (2,3)."""
        qc = QuantumCircuit(n_qubits)
        params = ParameterVector(prefix, length=n_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, n_qubits, 2), range(1, n_qubits, 2)):
            sub = self._pool_circuit(params[idx:idx+3], q1, q2)
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    def _pool_circuit(self, params, q1, q2):
        """Two‑qubit pooling unitary."""
        qc = QuantumCircuit(2)
        qc.cx(q1, q2)
        qc.rz(params[0], q2)
        qc.ry(params[1], q1)
        qc.cx(q1, q2)
        qc.rx(params[2], q1)
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the quantum network on a 2‑D input array.

        Parameters
        ----------
        data : np.ndarray
            Input image of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Expectation value of the observable.
        """
        flat = np.reshape(data, (1, self.n_qubits))
        # Binary encoding based on threshold
        param_bindings = {}
        for i, val in enumerate(flat[0]):
            param_bindings[self.feature_map.parameters[i]] = np.pi if val > self.threshold else 0.0

        # Fixed weight parameters (randomly initialized)
        weight_vals = np.zeros(len(self.qnn.weight_params))

        # Dummy inputs; actual values are supplied via param_bindings
        inputs = np.zeros((1, len(self.qnn.input_params)))

        # Evaluate
        result = self.qnn.evaluate(
            inputs=inputs,
            weight_vals=weight_vals,
            param_bindings=[param_bindings]
        )
        # The result is a 2‑D array; we take the first element
        return float(result[0][0])

def ConvFactory(kernel_size: int = 2, shots: int = 100,
                threshold: float = 127.0,
                backend: qiskit.providers.Backend = None) -> Conv:
    """
    Factory function for creating a :class:`Conv` instance.

    Returns
    -------
    Conv
        Configured quantum convolutional network.
    """
    return Conv(kernel_size=kernel_size, shots=shots,
                threshold=threshold, backend=backend)

__all__ = ["Conv", "ConvFactory"]
