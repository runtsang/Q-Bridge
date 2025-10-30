"""Quantum hybrid layer that mimics the classical ``HybridQuantumHybrid``.

The implementation is split into two sub‑circuits:

* ``QuanvCircuit`` – a parameterised convolution‑like circuit that
  operates on a 2‑D image patch.
* ``QuantumFC`` – a single‑qubit variational circuit that receives the
  parameter vector ``thetas``.

Both sub‑circuits are executed on the same Aer simulator and the
probabilities are combined linearly to produce a feature vector that
resembles the classical surrogate.

The API intentionally mirrors the ``HybridQuantumHybrid`` class so it
can be used as a drop‑in replacement in quantum‑aware workflows.
"""

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute
from typing import Iterable

class HybridQuantumHybrid:
    """Quantum implementation of the hybrid convolution + FC layer."""

    def __init__(
        self,
        kernel_size: int = 2,
        n_features: int = 1,
        conv_shots: int = 200,
        fc_shots: int = 200,
        conv_threshold: float = 0.5,
        fc_threshold: float = 0.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits_conv = kernel_size ** 2
        self.n_qubits_fc = 1
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Convolution circuit
        self.conv_circ = self._build_conv_circuit(
            self.n_qubits_conv, conv_shots, conv_threshold
        )

        # Fully connected circuit
        self.fc_circ = self._build_fc_circuit(self.n_qubits_fc, fc_shots, fc_threshold)

    def _build_conv_circuit(self, n_qubits, shots, threshold):
        circ = qiskit.QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            circ.rx(theta[i], i)
        circ.barrier()
        circ += random_circuit(n_qubits, 2)
        circ.measure_all()
        return {"circuit": circ, "shots": shots, "threshold": threshold, "theta": theta}

    def _build_fc_circuit(self, n_qubits, shots, threshold):
        circ = qiskit.QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        circ.h(range(n_qubits))
        circ.barrier()
        circ.ry(theta, range(n_qubits))
        circ.measure_all()
        return {"circuit": circ, "shots": shots, "threshold": threshold, "theta": theta}

    def run(self, image: np.ndarray, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the hybrid quantum layer.

        Parameters
        ----------
        image : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 1].
        thetas : Iterable[float]
            Parameter vector for the FC circuit.

        Returns
        -------
        np.ndarray
            Concatenated feature vector of shape (conv_channels + 1,).
        """
        # Convolution part
        conv_counts = self._run_conv(image)
        conv_features = conv_counts.mean(axis=0)

        # FC part
        fc_counts = self._run_fc(thetas)
        fc_features = fc_counts.squeeze()

        return np.concatenate([conv_features, fc_features])

    def _run_conv(self, image):
        n_qubits = self.n_qubits_conv
        circ = self.conv_circ["circuit"].copy()
        param_bindings = []
        for val in image.flatten():
            bind = {self.conv_circ["theta"][i]: np.pi if val > self.conv_circ["threshold"] else 0}
            param_bindings.append(bind)
        job = execute(
            circ,
            self.backend,
            shots=self.conv_circ["shots"],
            parameter_binds=param_bindings,
        )
        result = job.result()
        counts = result.get_counts(circ)
        probs = {k: v / self.conv_circ["shots"] for k, v in counts.items()}
        # Compute average number of |1> qubits per shot
        avg_ones = sum([sum(int(b) for b in key) * p for key, p in probs.items()])
        return np.full((1, n_qubits), avg_ones / n_qubits)

    def _run_fc(self, thetas):
        circ = self.fc_circ["circuit"].copy()
        param_bindings = [{self.fc_circ["theta"]: theta} for theta in thetas]
        job = execute(
            circ,
            self.backend,
            shots=self.fc_circ["shots"],
            parameter_binds=param_bindings,
        )
        result = job.result()
        counts = result.get_counts(circ)
        probs = {k: v / self.fc_circ["shots"] for k, v in counts.items()}
        # Expectation value of measurement in computational basis
        exp = sum([float(key) * p for key, p in probs.items()])
        return np.array([exp])

__all__ = ["HybridQuantumHybrid"]
