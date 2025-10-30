"""Quantum analog of ConvGen010 using Qiskit circuits."""

import qiskit
import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit.random import random_circuit

class ConvGen010:
    """
    Quantum implementation mirroring the classical ConvGen010 architecture.
    The class exposes a ``run`` method that accepts a 2‑D array and returns a scalar.
    It sequentially applies a quanvolution filter, a quantum self‑attention block,
    a parameterised EstimatorQNN, and a simple fully‑connected quantum circuit.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 127,
        attention_dim: int = 4,
        estimator_hidden: int = 8,
        fc_out_features: int = 1,
        shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.attention_dim = attention_dim
        self.estimator_hidden = estimator_hidden
        self.fc_out_features = fc_out_features
        self.shots = shots

        # Backend
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quanvolution circuit
        self.quanv = self._build_quanv()

        # Quantum self‑attention parameters
        self.rotation_params = [Parameter(f"rot_{i}") for i in range(attention_dim * 3)]
        self.entangle_params = [Parameter(f"ent_{i}") for i in range(attention_dim - 1)]
        self.attention_circuit = self._build_attention_circuit()

        # Estimator QNN
        self.estimator_qnn = self._build_estimator_qnn()

        # Fully connected quantum circuit
        self.fc_circuit = self._build_fc_circuit()

    def _build_quanv(self):
        n_qubits = self.kernel_size ** 2
        qc = qiskit.QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc

    def _build_attention_circuit(self):
        n_qubits = self.attention_dim
        qc = qiskit.QuantumCircuit(n_qubits)
        # Rotation gates
        for i in range(n_qubits):
            qc.rx(self.rotation_params[3 * i], i)
            qc.ry(self.rotation_params[3 * i + 1], i)
            qc.rz(self.rotation_params[3 * i + 2], i)
        # Entangling gates
        for i in range(n_qubits - 1):
            qc.crx(self.entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def _build_estimator_qnn(self):
        params1 = [Parameter("input1"), Parameter("weight1")]
        qc1 = qiskit.QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)
        observable1 = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        estimator_qnn = EstimatorQNN(
            circuit=qc1,
            observables=observable1,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=estimator,
        )
        return estimator_qnn

    def _build_fc_circuit(self):
        qc = qiskit.QuantumCircuit(1)
        theta = Parameter("theta")
        qc.h(0)
        qc.ry(theta, 0)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the full quantum pipeline on a 2‑D array.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).

        Returns:
            Scalar output after the fully‑connected layer.
        """
        # Quanvolution
        data_flat = data.reshape(1, self.kernel_size ** 2)
        param_binds = []
        for dat in data_flat:
            bind = {theta: np.pi if val > self.conv_threshold else 0
                    for theta, val in zip(self.quanv.parameters, dat)}
            param_binds.append(bind)
        job = qiskit.execute(self.quanv, self.backend,
                             shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.quanv)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        quanv_out = counts / (self.shots * self.kernel_size ** 2)

        # Quantum self‑attention
        rotation_binds = {p: np.random.rand() for p in self.rotation_params}
        entangle_binds = {p: np.random.rand() for p in self.entangle_params}
        job = qiskit.execute(self.attention_circuit, self.backend,
                             shots=self.shots,
                             parameter_binds=[rotation_binds, entangle_binds])
        attn_counts = job.result().get_counts(self.attention_circuit)
        attn_out = sum(int(key, 2) * val for key, val in attn_counts.items()) / (
            self.shots * self.attention_dim
        )

        # Estimator QNN
        est_output = self.estimator_qnn.predict(
            {self.estimator_qnn.input_params[0]: attn_out}
        )[0]

        # Fully connected quantum circuit
        fc_bind = {self.fc_circuit.parameters[0]: est_output}
        job = qiskit.execute(self.fc_circuit, self.backend,
                             shots=self.shots, parameter_binds=[fc_bind])
        fc_counts = job.result().get_counts(self.fc_circuit)
        fc_out = sum(int(key, 2) * val for key, val in fc_counts.items()) / (
            self.shots * self.fc_out_features
        )

        return fc_out

__all__ = ["ConvGen010"]
