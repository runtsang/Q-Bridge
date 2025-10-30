from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class QuantumHybridConvEstimator:
    """
    Quantum implementation of the hybrid conv estimator.
    Uses a parameterised conv circuit for each patch and a
    qiskit EstimatorQNN for regression.
    """

    def __init__(
        self,
        kernel_sizes: list[int] = [2, 3],
        threshold: int = 127,
        shots: int = 1024,
    ) -> None:
        self.kernel_sizes = kernel_sizes
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Build quantum conv circuits for each kernel
        self.conv_circuits = {}
        for k in kernel_sizes:
            n_qubits = k * k
            qc = QuantumCircuit(n_qubits)
            theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
            for i in range(n_qubits):
                qc.rx(theta[i], i)
            qc.barrier()
            qc += random_circuit(n_qubits, 2)
            self.conv_circuits[k] = (qc, theta)

        # Build a simple EstimatorQNN to map conv outputs to a scalar
        self.estimator_qnn = self._build_estimator_qnn()

    def _build_estimator_qnn(self):
        # Single qubit circuit that takes the mean of conv outputs
        qc = QuantumCircuit(1)
        inp = Parameter("inp")
        weight = Parameter("wgt")
        qc.h(0)
        qc.ry(inp, 0)
        qc.rx(weight, 0)

        observable = SparsePauliOp.from_list([("Y", 1)])
        est = StatevectorEstimator()
        return EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[inp],
            weight_params=[weight],
            estimator=est,
        )

    def _run_conv_circuit(self, qc: QuantumCircuit, theta: list[Parameter], data: np.ndarray) -> float:
        """Run a single conv circuit on a flattened patch."""
        param_bind = {}
        for i, val in enumerate(data):
            param_bind[theta[i]] = np.pi if val > self.threshold else 0
        job = execute(
            qc,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(qc)
        # Compute average probability of measuring |1>
        prob_one = 0.0
        for bitstring, count in counts.items():
            prob_one += (bitstring.count("1") * count)
        total = self.shots * qc.num_qubits
        return prob_one / total

    def run(self, image: np.ndarray) -> float:
        """
        Run the hybrid quantum estimator on a grayscale image.

        Parameters
        ----------
        image : np.ndarray
            2D array of pixel intensities.

        Returns
        -------
        float
            Estimated scalar output.
        """
        conv_features = []
        for k in self.kernel_sizes:
            stride = k
            for i in range(0, image.shape[0] - k + 1, stride):
                for j in range(0, image.shape[1] - k + 1, stride):
                    patch = image[i : i + k, j : j + k].flatten()
                    qc, theta = self.conv_circuits[k]
                    val = self._run_conv_circuit(qc, theta, patch)
                    conv_features.append(val)

        # Convert to a single input for the EstimatorQNN
        if not conv_features:
            conv_features = [0.0]
        input_val = np.mean(conv_features)

        # Estimate weight via the QNN
        output = self.estimator_qnn({"inp": input_val, "wgt": 0.0}).real
        return output

__all__ = ["QuantumHybridConvEstimator"]
