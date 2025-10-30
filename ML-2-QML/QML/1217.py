"""Variational quanvolution layer with trainable parameters.

The public API mirrors the original ``Conv`` seed: a callable
``ConvGen158()`` that returns an object exposing a ``run`` method.
The quantum circuit now contains a trainable RX rotation per qubit
followed by a fixed data‑dependent rotation, allowing gradient‑based
optimisation of the filter weights.  The circuit is executed on a
Qiskit Aer simulator.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit

__all__ = ["ConvGen158"]


def ConvGen158(kernel_size: int = 2,
               backend: qiskit.providers.BaseBackend | None = None,
               shots: int = 1024,
               threshold: float = 127.0,
               trainable: bool = True) -> object:
    """
    Return a variational quanvolution filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter (number of qubits = kernel_size**2).
    backend : qiskit.providers.BaseBackend, optional
        Qiskit backend; if None the Aer qasm simulator is used.
    shots : int, default 1024
        Number of shots for each execution.
    threshold : float, default 127.0
        Threshold used to encode classical input into a rotation of π.
    trainable : bool, default True
        If True the filter contains a trainable RX rotation per qubit
        that can be optimised by gradient‑based methods.
    """

    class QuanvCircuit:
        """Variational quanvolution circuit."""

        def __init__(self, kernel_size: int, backend, shots: int, threshold: float,
                     trainable: bool) -> None:
            self.n_qubits = kernel_size ** 2
            self.backend = backend or Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.threshold = threshold
            self.trainable = trainable

            # Data‑dependent parameters (fixed during training)
            self.data_params = [Parameter(f"data_{i}") for i in range(self.n_qubits)]

            # Trainable parameters
            self.train_params = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]

            # Current numerical values of trainable params (initialised to 0)
            self.param_values = {p: 0.0 for p in self.train_params}

            # Build the circuit template
            self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
            # Data‑dependent rotations
            for i in range(self.n_qubits):
                self.circuit.rx(self.data_params[i], i)
            # Trainable rotations
            for i in range(self.n_qubits):
                self.circuit.rx(self.train_params[i], i)
            # Entangling layer (simple chain of CNOTs)
            for i in range(self.n_qubits - 1):
                self.circuit.cx(i, i + 1)
            # Optional random circuit to add expressivity
            self.circuit += random_circuit(self.n_qubits, 2)
            # Measurement
            self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

        def set_trainable_params(self, values: dict[Parameter, float]) -> None:
            """
            Update the numerical values of the trainable parameters.

            Parameters
            ----------
            values : dict
                Mapping from Parameter objects to float values.
            """
            for p, val in values.items():
                if p in self.param_values:
                    self.param_values[p] = float(val)

        def run(self, data: np.ndarray) -> float:
            """
            Execute the circuit on the supplied data.

            Parameters
            ----------
            data : np.ndarray
                2‑D array with shape (kernel_size, kernel_size) containing
                integer pixel values.

            Returns
            -------
            float
                Average probability of measuring |1> across all qubits.
            """
            flat = data.reshape(1, self.n_qubits)
            param_binds = []
            for row in flat:
                bind = {}
                # Data‑dependent rotation: π if value > threshold else 0
                for i, val in enumerate(row):
                    bind[self.data_params[i]] = np.pi if val > self.threshold else 0.0
                # Trainable rotation values
                for p in self.train_params:
                    bind[p] = self.param_values[p]
                param_binds.append(bind)

            job = execute(self.circuit,
                          backend=self.backend,
                          shots=self.shots,
                          parameter_binds=param_binds)
            result = job.result()
            counts = result.get_counts(self.circuit)

            # Compute average probability of |1> over all qubits
            total_ones = 0
            total_shots = self.shots * len(param_binds)
            for bitstring, freq in counts.items():
                ones = bitstring.count("1")
                total_ones += ones * freq

            return total_ones / (total_shots * self.n_qubits)

        def get_trainable_params(self) -> list[Parameter]:
            """Return the list of trainable parameters."""
            return self.train_params

    return QuanvCircuit(kernel_size, backend, shots, threshold, trainable)
