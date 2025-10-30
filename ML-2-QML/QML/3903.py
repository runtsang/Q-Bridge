"""Hybrid Sampler Quantum Neural Network – Quantum implementation.

Implements a 2‑qubit variational circuit that mimics the classical
HybridSamplerQNN.  Input parameters control the Ry rotations,
while weight parameters are learned by a classical optimiser.
The circuit is run on a state‑vector simulator and returns
the expectation value of Z on qubit 0, as well as a probability
distribution over the computational basis.  A small wrapper
provides a `sample` method that draws measurement results.
"""

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
import numpy as np


class HybridSamplerQNN:
    """
    Quantum hybrid sampler.

    Parameters
    ----------
    backend : qiskit.providers.Provider, optional
        Quantum backend; defaults to Aer state‑vector simulator.
    shots : int, optional
        Number of shots for sampling; defaults to 1024.
    """

    def __init__(self, backend=None, shots: int = 1024) -> None:
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.shots = shots
        # Parameter vectors
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", 4)
        # Build circuit
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Input rotations
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)
        # Weight rotations
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weight_params[2], 0)
        qc.ry(self.weight_params[3], 1)
        return qc

    def expectation(self, input_vals: np.ndarray, weight_vals: np.ndarray) -> np.ndarray:
        """
        Compute expectation value of Pauli‑Z on qubit 0.

        Args:
            input_vals: array-like of shape (2,) – input parameters.
            weight_vals: array-like of shape (4,) – weight parameters.

        Returns:
            Expectation value as a numpy array of shape (1,).
        """
        param_dict = {
            **{p: v for p, v in zip(self.input_params, input_vals)},
            **{p: v for p, v in zip(self.weight_params, weight_vals)},
        }
        bound_qc = self.circuit.bind_parameters(param_dict)
        state = execute(bound_qc, self.backend).result().get_statevector()
        # Expectation of Z on qubit 0: sum_p (-1)^{bit0} |p|^2
        probs = np.abs(state) ** 2
        bit0 = (np.arange(len(probs)) >> 1) & 1
        expectation = np.sum((1 - 2 * bit0) * probs)
        return np.array([expectation])

    def sample(self, input_vals: np.ndarray, weight_vals: np.ndarray,
               n_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the circuit.

        Args:
            input_vals: array-like of shape (2,) – input parameters.
            weight_vals: array-like of shape (4,) – weight parameters.
            n_samples: number of shots to simulate.

        Returns:
            Numpy array of shape (n_samples, 2) – one‑hot encoded measurement results.
        """
        param_dict = {
            **{p: v for p, v in zip(self.input_params, input_vals)},
            **{p: v for p, v in zip(self.weight_params, weight_vals)},
        }
        bound_qc = self.circuit.bind_parameters(param_dict)
        job = execute(bound_qc, Aer.get_backend("qasm_simulator"),
                      shots=n_samples)
        counts = job.result().get_counts(bound_qc)
        samples = np.array(list(counts.keys()), dtype=int)
        probs = np.array(list(counts.values())) / n_samples
        # Convert integer bitstrings to one‑hot vectors
        one_hot = np.zeros((n_samples, 2), dtype=int)
        for idx, prob in zip(samples, probs):
            bits = np.array(list(f"{idx:02b}"), dtype=int)
            one_hot += int(prob * n_samples) * bits[::-1]  # reverse for qubit ordering
        return one_hot

    def draw(self) -> None:
        """Visualise the variational circuit."""
        self.circuit.draw("mpl")
