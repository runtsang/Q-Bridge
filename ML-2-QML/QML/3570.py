"""
ConvGen198 – Quantum implementation combining a quanvolution layer
and a fully‑connected parameterised circuit.

The design follows the structure from the QML Conv and FCL seeds,
adding a clear synergy: data is encoded into a quantum convolution
circuit, which outputs a probability.  That probability is then
combined with a separate quantum fully‑connected circuit that
produces a single expectation value.  The two results are returned
as a tuple for downstream classical processing.
"""

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit


class ConvGen198:
    """
    Hybrid quantum layer that first runs a quanvolution circuit on the
    input data and then evaluates a parameterised fully‑connected circuit.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        fc_qubits: int = 1,
        backend: qiskit.providers.BaseBackend | None = None,
        shots: int = 100,
        conv_threshold: float = 127.0,
    ) -> None:
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")

        # Quanvolution (data‑dependent) circuit
        self.conv_circuit = self._build_quanv_circuit(
            conv_kernel_size, backend, shots, conv_threshold
        )

        # Fully‑connected quantum circuit (parameterized by thetas)
        self.fc_circuit = self._build_fc_circuit(
            fc_qubits, backend, shots
        )

    def _build_quanv_circuit(
        self,
        kernel_size: int,
        backend: qiskit.providers.BaseBackend,
        shots: int,
        threshold: float,
    ) -> qiskit.QuantumCircuit:
        n_qubits = kernel_size ** 2
        qc = qiskit.QuantumCircuit(n_qubits)
        theta_params = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i, th in enumerate(theta_params):
            qc.rx(th, i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()

        self._conv_backend = backend
        self._conv_shots = shots
        self._conv_threshold = threshold
        self._conv_params = theta_params
        return qc

    def _build_fc_circuit(
        self,
        n_qubits: int,
        backend: qiskit.providers.BaseBackend,
        shots: int,
    ) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        qc.h(range(n_qubits))
        qc.barrier()
        qc.ry(theta, range(n_qubits))
        qc.measure_all()

        self._fc_backend = backend
        self._fc_shots = shots
        self._fc_param = theta
        return qc

    def run(self, data: np.ndarray, thetas: list[float]) -> tuple[float, float]:
        """
        Execute the hybrid quantum pipeline.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 255].
        thetas : list[float]
            Parameters for the fully‑connected circuit.

        Returns
        -------
        tuple[float, float]
            (conv_output, fc_output) where conv_output is the average
            probability of measuring |1> across the quanvolution qubits,
            and fc_output is the expectation value of the fully‑connected
            circuit.
        """
        # Quanvolution run
        data_flat = np.reshape(data, (1, self._conv_circuit.num_qubits))
        conv_param_binds = []
        for row in data_flat:
            bind = {
                p: np.pi if val > self._conv_threshold else 0
                for p, val in zip(self._conv_params, row)
            }
            conv_param_binds.append(bind)

        conv_job = qiskit.execute(
            self._conv_circuit,
            backend=self._conv_backend,
            shots=self._conv_shots,
            parameter_binds=conv_param_binds,
        )
        conv_counts = conv_job.result().get_counts(self._conv_circuit)

        conv_total = 0
        for key, val in conv_counts.items():
            ones = sum(int(bit) for bit in key)
            conv_total += ones * val
        conv_output = conv_total / (self._conv_shots * self._conv_circuit.num_qubits)

        # Fully‑connected run
        fc_job = qiskit.execute(
            self._fc_circuit,
            backend=self._fc_backend,
            shots=self._fc_shots,
            parameter_binds=[{self._fc_param: theta} for theta in thetas],
        )
        fc_counts = fc_job.result().get_counts(self._fc_circuit)

        fc_counts_arr = np.array(list(fc_counts.values()))
        fc_states = np.array([int(k, 2) for k in fc_counts.keys()], dtype=float)
        fc_probs = fc_counts_arr / self._fc_shots
        fc_expectation = np.sum(fc_states * fc_probs) / (2 ** self._fc_circuit.num_qubits)

        return conv_output, float(fc_expectation)


__all__ = ["ConvGen198"]
