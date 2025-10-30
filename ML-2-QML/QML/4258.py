import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class HybridConvFraudLayer:
    """
    Quantum hybrid layer that chains a quanvolution (quantum convolution) with a
    parameterised fully‑connected quantum circuit.  The convolution processes
    the raw 2‑D image patch, while the fully‑connected stage maps the resulting
    probability to a final expectation value.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 100,
        threshold: float = 127.0,
        backend: qiskit.providers.Backend | None = None,
        clip: bool = True,
    ) -> None:
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.clip = clip

        # Quantum convolution
        self.n_qubits = kernel_size ** 2
        self._conv_circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._conv_circuit.rx(self.theta[i], i)
        self._conv_circuit.barrier()
        self._conv_circuit += random_circuit(self.n_qubits, 2)
        self._conv_circuit.measure_all()

        # Quantum fully‑connected layer
        self.n_fc_qubits = 1
        self._fc_circuit = qiskit.QuantumCircuit(self.n_fc_qubits)
        self.fc_theta = qiskit.circuit.Parameter("theta_fc")
        self._fc_circuit.h(range(self.n_fc_qubits))
        self._fc_circuit.barrier()
        self._fc_circuit.ry(self.fc_theta, range(self.n_fc_qubits))
        self._fc_circuit.measure_all()

    def _run_conv(self, data: np.ndarray) -> float:
        """
        Execute the quanvolution circuit on a 2‑D patch and return the average
        probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = qiskit.execute(
            self._conv_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._conv_circuit)
        ones = 0
        for key, val in result.items():
            ones += sum(int(bit) for bit in key) * val
        return ones / (self.shots * self.n_qubits)

    def _run_fc(self, conv_output: float) -> float:
        """
        Run the fully‑connected quantum circuit with the convolution output as
        the rotation angle.
        """
        job = qiskit.execute(
            self._fc_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.fc_theta: conv_output}],
        )
        result = job.result().get_counts(self._fc_circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return expectation

    def run(self, data: np.ndarray) -> float:
        """
        Execute the full quantum hybrid pipeline.

        Parameters
        ----------
        data
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Final expectation value after the fully‑connected stage.
        """
        conv_out = self._run_conv(data)
        return self._run_fc(conv_out)

    def __call__(self, data: np.ndarray) -> float:
        return self.run(data)

__all__ = ["HybridConvFraudLayer"]
