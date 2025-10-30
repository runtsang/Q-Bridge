from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.providers.aer import AerSimulator
from qiskit import execute

class Conv__gen438:
    """
    Quantum‑centric hybrid convolutional filter that emulates the original
    Conv filter but uses a Qiskit variational circuit as the quantum layer.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 100,
        threshold: float = 127,
        fraud_params: dict | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots

        # Build the parameterised circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = AerSimulator()
        self.fraud_params = fraud_params or {
            "bs_theta": 1.0,
            "bs_phi": 0.0,
            "phases": (0.0, 0.0),
            "squeeze_r": (0.0, 0.0),
            "squeeze_phi": (0.0, 0.0),
            "displacement_r": (1.0, 1.0),
            "displacement_phi": (0.0, 0.0),
            "shift": (0.0, 0.0),
        }

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum convolutional filter on a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            Input image patch of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar output of the filter.
        """
        flat = data.reshape(self.n_qubits)
        param_binds = []
        for i, val in enumerate(flat):
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0}
            param_binds.append(bind)

        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self._circuit)

        total_ones = 0
        total_counts = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count('1')
            total_ones += ones * freq
            total_counts += freq
        prob_one = total_ones / (self.shots * self.n_qubits)

        # Apply fraud regularisation
        return self._apply_fraud(prob_one)

    def _apply_fraud(self, value: float) -> float:
        """
        Simple fraud‑regularisation that emulates the photonic fraud block.
        """
        vec = np.array([value, value], dtype=np.float32)

        weight = np.array([[self.fraud_params["bs_theta"], self.fraud_params["bs_phi"]],
                           [self.fraud_params["squeeze_r"][0], self.fraud_params["squeeze_r"][1]]],
                          dtype=np.float32)
        bias = np.array(self.fraud_params["phases"], dtype=np.float32)

        out = vec @ weight.T + bias
        out = np.tanh(out)

        scale = np.array(self.fraud_params["displacement_r"], dtype=np.float32)
        shift = np.array(self.fraud_params["displacement_phi"], dtype=np.float32)

        out = out * scale + shift
        return out.mean()
