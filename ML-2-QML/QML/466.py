"""ConvGen: a multi‑scale quantum convolutional filter with parameter‑shift gradients.

The quantum implementation mirrors the classical ConvGen but replaces each
depth‑wise filter with a small variational circuit.  For each kernel size
a separate circuit is constructed.  The circuits are executed on a Qiskit
simulator, and the average probability of measuring |1> is used as the
activation.  A simple linear fusion is applied to combine the per‑kernel
outputs.  The class also exposes a `gradient` method that estimates the
gradient of the output w.r.t. each rotation angle using the
parameter‑shift rule, enabling hybrid back‑propagation with a classical
optimizer.
"""

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute
from qiskit.providers import Backend
from typing import List, Dict, Tuple


class ConvGen:
    def __init__(
        self,
        kernel_sizes: List[int] | int = 2,
        backend: Backend | None = None,
        shots: int = 100,
        threshold: float = 127.0,
        trainable: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        kernel_sizes : list or int
            List of kernel sizes or a single integer.
        backend : qiskit provider backend
            Quantum backend to execute circuits.  If None, uses Aer qasm_simulator.
        shots : int
            Number of shots per circuit execution.
        threshold : float
            Threshold used to set rotation angles.
        trainable : bool
            Whether the rotation angles are trainable parameters.
        """
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        self.kernel_sizes = kernel_sizes
        self.shots = shots
        self.threshold = threshold
        self.trainable = trainable

        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuits: Dict[int, QuantumCircuit] = {}
        self.theta: Dict[int, List[Parameter]] = {}

        for ks in self.kernel_sizes:
            n_qubits = ks * ks
            qc = QuantumCircuit(n_qubits, n_qubits)
            thetas = [Parameter(f"theta_{ks}_{i}") for i in range(n_qubits)]
            self.theta[ks] = thetas
            for i, th in enumerate(thetas):
                qc.rx(th, i)
            qc += random_circuit(n_qubits, depth=2, seed=42)
            qc.measure(range(n_qubits), range(n_qubits))
            self.circuits[ks] = qc

        # Fusion weights: one weight per kernel size
        self.fusion_weights = np.ones(len(self.kernel_sizes)) / len(self.kernel_sizes)

    def _bind_parameters(self, qc: QuantumCircuit, data: np.ndarray, ks: int) -> QuantumCircuit:
        """
        Bind rotation angles based on data and threshold.
        """
        n_qubits = ks * ks
        bind_dict = {}
        for i in range(n_qubits):
            val = data.ravel()[i]
            angle = np.pi if val > self.threshold else 0.0
            bind_dict[self.theta[ks][i]] = angle
        return qc.bind_parameters(bind_dict)

    def _execute_circuit(self, qc: QuantumCircuit) -> float:
        """
        Execute a circuit and return the average probability of measuring |1>.
        """
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)
        # Compute average |1> probability across all qubits
        total_ones = 0
        total_counts = 0
        for bitstring, count in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * count
            total_counts += count
        avg_prob = total_ones / (total_counts * len(qc))
        return avg_prob

    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the quantum convolution on a single 2‑D sample.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (H, W) with values in [0, 255].

        Returns
        -------
        float
            Fused activation value across all kernel sizes.
        """
        outputs = []
        for idx, ks in enumerate(self.kernel_sizes):
            if data.shape[0] < ks or data.shape[1] < ks:
                raise ValueError(f"Data size {data.shape} too small for kernel {ks}")
            sub = data[:ks, :ks]
            qc = self.circuits[ks]
            bound_qc = self._bind_parameters(qc, sub, ks)
            prob = self._execute_circuit(bound_qc)
            outputs.append(prob)

        outputs = np.array(outputs)
        fused = np.dot(outputs, self.fusion_weights)
        return float(fused)

    def parameter_shift_gradient(self, data: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Estimate gradient of the output w.r.t. each rotation angle using
        the parameter‑shift rule.  Returns a dictionary mapping kernel size
        to a gradient array of shape (ks*ks,).
        """
        grads: Dict[int, np.ndarray] = {}
        for ks in self.kernel_sizes:
            n_qubits = ks * ks
            grad = np.zeros(n_qubits)
            for i in range(n_qubits):
                shift = np.pi / 2
                # + shift
                bind_plus = {}
                for j in range(n_qubits):
                    angle = np.pi if data.ravel()[j] > self.threshold else 0.0
                    if j == i:
                        angle += shift
                    bind_plus[self.theta[ks][j]] = angle
                qc_plus = self.circuits[ks].bind_parameters(bind_plus)
                prob_plus = self._execute_circuit(qc_plus)

                # - shift
                bind_minus = {}
                for j in range(n_qubits):
                    angle = np.pi if data.ravel()[j] > self.threshold else 0.0
                    if j == i:
                        angle -= shift
                    bind_minus[self.theta[ks][j]] = angle
                qc_minus = self.circuits[ks].bind_parameters(bind_minus)
                prob_minus = self._execute_circuit(qc_minus)

                grad[i] = 0.5 * (prob_plus - prob_minus)
            grads[ks] = grad
        return grads

__all__ = ["ConvGen"]
