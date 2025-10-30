"""Hybrid quantum convolution + estimator.

This module implements a quantum‑classical hybrid layer that mirrors the
classical `HybridConvEstimator`.  The quantum part replaces the
convolutional filter with a parameterised circuit that encodes the
pixel values as rotations and entangles them.  The output expectation
value is fed into a Qiskit EstimatorQNN that performs a regression
over the measured expectation.

Key points
----------
* `QuanvCircuit` – a reusable filter circuit that accepts a
  threshold for pixel activation.
* `HybridConvEstimator` – exposes the same API as the classical
  counterpart but internally uses Qiskit primitives.
* The implementation uses the Aer `qasm_simulator` as backend and
  defaults to 200 shots for a balance between speed and accuracy.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QML_EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from typing import Tuple


class QuanvCircuit:
    """
    Quantum convolutional filter that maps a 2‑D kernel into a
    probability of measuring |1⟩ across all qubits.

    The circuit is composed of:
    * an RX rotation per qubit encoding the pixel value,
    * a shallow entangling layer,
    * measurement of all qubits.
    """

    def __init__(self, kernel_size: int, threshold: float, shots: int = 200):
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        self.theta = ParameterVector("theta", self.n_qubits)
        self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)

        # Encode pixel values
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)

        # Entangle
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()

        # Measurement
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def run(self, data: np.ndarray) -> float:
        """
        Execute the filter on a single kernel.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with entries
            in [0, 255].

        Returns
        -------
        float
            Normalised count of |1⟩ outcomes averaged over all qubits.
        """
        n = self.n_qubits
        data_flat = data.reshape(1, n)
        param_binds = []

        for sample in data_flat:
            bind = {}
            for idx, val in enumerate(sample):
                bind[self.theta[idx]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        ones = 0
        for bitstring, freq in counts.items():
            ones += bitstring.count("1") * freq

        return ones / (self.shots * n)


class HybridConvEstimator:
    """
    Quantum‑classical hybrid version of the classical
    `HybridConvEstimator`.  The convolution is performed by
    `QuanvCircuit` and the regression by a Qiskit EstimatorQNN.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127.0,
        shots: int = 200,
        regressor_hidden: Tuple[int,...] = (8, 4),
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots

        # Quantum convolution filter
        self.filter = QuanvCircuit(kernel_size, threshold, shots)

        # Classical estimator for regression
        # Create a simple quantum circuit with a single qubit
        qc = QuantumCircuit(1)
        qc.h(0)

        # Parameters for input and weight
        input_param = Parameter("input")
        weight_param = Parameter("weight")

        # Build a circuit that applies Ry(input) then Rx(weight)
        qc.ry(input_param, 0)
        qc.rx(weight_param, 0)

        # Observable Y
        from qiskit.quantum_info import SparsePauliOp

        observable = SparsePauliOp.from_list([("Y", 1)])

        # Assemble EstimatorQNN
        estimator = StatevectorEstimator()
        self.estimator_qnn = QML_EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=estimator,
        )

        # Initialise weight parameter to zero
        self.estimator_qnn.set_weights([0.0])

    def run(self, data: np.ndarray) -> float:
        """
        Forward pass through the quantum filter and the EstimatorQNN.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Regression output from the EstimatorQNN.
        """
        # Convolution filter output
        conv_value = self.filter.run(data)

        # Predict using EstimatorQNN; the input is a single float
        prediction = self.estimator_qnn.predict([conv_value])[0]
        return float(prediction)

    def set_weights(self, weights: list[float]) -> None:
        """Update the EstimatorQNN weight."""
        self.estimator_qnn.set_weights(weights)

    def get_weights(self) -> list[float]:
        """Retrieve the current EstimatorQNN weight."""
        return self.estimator_qnn.get_weights()


__all__ = ["HybridConvEstimator"]
