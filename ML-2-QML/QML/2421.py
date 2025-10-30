"""Hybrid estimator that fuses a quantum convolutional filter with a quantum neural network.

The class exposes a `run` method that accepts a 2‑D array of shape
(k, k) and returns a scalar regression output.  Internally it uses a
`QuanvCircuit` to encode the image patch as rotation angles, then
measures the probability of observing |1>.  The resulting scalar is
fed into an `EstimatorQNN` that maps it to a regression value.  The
design is fully self‑contained and can be used with any Qiskit backend.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from typing import Any

__all__ = ["HybridEstimator"]


class QuanvCircuit:
    """
    Quantum analogue of a convolutional filter (quanvolution).

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel.
    backend : qiskit.providers.BaseBackend
        Backend on which to execute the circuit.
    shots : int
        Number of shots for sampling.
    threshold : float
        Threshold to decide the rotation angle (π or 0).
    """

    def __init__(
        self,
        kernel_size: int,
        backend: Any,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        # Rotation angles are treated as parameters; they will be bound later.
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: Any) -> float:
        """
        Execute the circuit on the provided image patch and return the
        average probability of measuring |1> across all qubits.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Expected probability of observing |1>.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


class HybridEstimator:
    """
    Hybrid quantum estimator that first extracts a feature via a quanvolution
    circuit and then applies a quantum neural network to produce a scalar
    regression output.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    threshold : float, default 127
        Threshold used when encoding the input into rotation angles.
    shots : int, default 100
        Number of shots for the quanvolution circuit.
    backend : qiskit.providers.BaseBackend | None, default None
        Backend for executing the circuits.  If None, the Aer qasm_simulator
        is used.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        shots: int = 100,
        backend: Any | None = None,
    ) -> None:
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.quantum_conv = QuanvCircuit(kernel_size, backend, shots, threshold)

        # Define the small variational circuit used by EstimatorQNN
        input_param = Parameter("input_param")
        weight_param = Parameter("weight_param")
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.ry(input_param, 0)
        circuit.rx(weight_param, 0)

        observable = SparsePauliOp.from_list([("Y", 1)])

        estimator = StatevectorEstimator()
        self.estimator_qnn = QEstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=estimator,
        )

        # Store parameters for later binding
        self._input_param = input_param
        self._weight_param = weight_param

    def run(self, data: Any) -> float:
        """
        Execute a full forward pass.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Regression output produced by the quantum neural network.
        """
        # Feature extraction by the quanvolution circuit
        feature = self.quantum_conv.run(data)

        # Bind the extracted feature to the input parameter of EstimatorQNN.
        # The weight parameter is left at its initial value (0.0) but can be
        # treated as a trainable hyper‑parameter if desired.
        param_dict = {self._input_param: feature, self._weight_param: 0.0}
        result = self.estimator_qnn.evaluate(param_dict)
        # The estimator returns a list of expectation values; we take the first.
        return float(result[0])

    # Compatibility shim for the original EstimatorQNN function
    @staticmethod
    def EstimatorQNN() -> "HybridEstimator":
        return HybridEstimator()
