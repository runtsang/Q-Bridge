from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

class HybridEstimatorQNN:
    """
    Hybrid quantum estimator that combines a quanvolution filter with a
    variational EstimatorQNN.

    The filter is a parameterised quantum circuit that encodes classical
    data into qubit rotations; the output probability is used as a classical
    input to the EstimatorQNN.  The network remains trainable via classical
    optimisation of the circuit weights.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        # Quanv filter
        self.filter = self._build_filter(kernel_size, threshold, shots)
        # EstimatorQNN circuit
        self.estimator_circuit, self.estimator_qnn = self._build_estimator()

    def _build_filter(self, kernel_size: int, threshold: float, shots: int):
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i, p in enumerate(theta):
            qc.rx(p, i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        backend = qiskit.Aer.get_backend("qasm_simulator")
        return {
            "circuit": qc,
            "theta": theta,
            "backend": backend,
            "shots": shots,
            "threshold": threshold,
        }

    def _build_estimator(self):
        # EstimatorQNN expects a circuit with input and weight parameters.
        input_param = Parameter("input")
        weight_param = Parameter("weight")
        # Simple circuit: single qubit with Ry(input) and Rz(weight)
        circuit = QuantumCircuit(1)
        circuit.ry(input_param, 0)
        circuit.rz(weight_param, 0)
        circuit.measure_all()
        # Observable Y on the single qubit
        from qiskit.quantum_info import SparsePauliOp
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = Estimator()
        estimator_qnn = QEstimatorQNN(
            circuit=circuit,
            observables=observable,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=estimator,
        )
        return circuit, estimator_qnn

    def evaluate(self, data: np.ndarray, weight: float = 0.0) -> float:
        """
        Run the hybrid quantum estimator on a 2‑D input array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        weight : float
            Classical weight to bind to the EstimatorQNN circuit.

        Returns
        -------
        float
            Expectation value from the EstimatorQNN circuit.
        """
        # Run filter
        bind = {}
        for i, val in enumerate(data.flatten()):
            bind[self.filter["theta"][i]] = np.pi if val > self.filter["threshold"] else 0
        job = qiskit.execute(
            self.filter["circuit"],
            self.filter["backend"],
            shots=self.filter["shots"],
            parameter_binds=[bind],
        )
        result = job.result().get_counts(self.filter["circuit"])
        # Compute average probability of measuring |1> across qubits
        counts = sum(sum(int(bit) for bit in key) * val for key, val in result.items())
        prob = counts / (self.filter["shots"] * self.filter["circuit"].num_qubits)

        # Bind the probability as input to the EstimatorQNN
        input_bind = {self.estimator_qnn.input_params[0]: prob}
        weight_bind = {self.estimator_qnn.weight_params[0]: weight}
        value = self.estimator_qnn.run(
            self.estimator_circuit,
            input_bind,
            weight_bind,
            backend=self.estimator_qnn.estimator.backend,
        )
        return value

__all__ = ["HybridEstimatorQNN"]
