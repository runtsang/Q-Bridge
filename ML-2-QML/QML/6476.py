"""Hybrid quantum module combining a fully connected + sampler network.

The class builds a parameterized quantum circuit that mirrors the
classical `HybridFCLSampler` structure:

* A 2‑qubit circuit implements the sampler QNN (Ry‑CX‑Ry‑CX‑Ry pattern).
* An additional Ry rotation on qubit 0 represents the fully connected
  layer weight.

The `run` method accepts a dictionary of parameters and returns an
expectation value computed from measurement statistics.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector


class HybridFCLSamplerQuantum:
    """Quantum implementation of the hybrid fully‑connected + sampler network.

    Parameters
    ----------
    n_qubits : int, default=2
        Number of qubits in the circuit.
    backend : qiskit.providers.BaseBackend, optional
        Execution backend; defaults to Aer qasm simulator.
    shots : int, default=1024
        Number of shots for sampling.
    """

    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Parameter vectors
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)
        self.fc_weight = ParameterVector("fc_weight", 1)

        # Build circuit
        self.circuit = QuantumCircuit(n_qubits)
        # Sampler QNN part
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)
        # Fully connected layer rotation on qubit 0
        self.circuit.ry(self.fc_weight[0], 0)
        self.circuit.measure_all()

    def run(self, params: dict) -> np.ndarray:
        """Execute the circuit with the supplied parameters.

        Parameters
        ----------
        params
            Dictionary with keys 'inputs', 'weights', 'fc_weight'.  Each
            value should be an iterable of floats matching the
            corresponding ParameterVector length.

        Returns
        -------
        np.ndarray
            Expectation value computed from the measurement histogram.
        """
        # Bind parameters
        binds = [{p: v for p, v in zip(self.inputs, params["inputs"])},
                 {p: v for p, v in zip(self.weights, params["weights"])},
                 {p: v for p, v in zip(self.fc_weight, params["fc_weight"])}]
        bound_circuit = self.circuit.bind_parameters(binds[0])
        bound_circuit = bound_circuit.bind_parameters(binds[1])
        bound_circuit = bound_circuit.bind_parameters(binds[2])

        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])

        # Expectation: encode bitstring as integer value
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def draw(self, style: str = "mpl") -> None:
        """Print a visual representation of the circuit."""
        print(self.circuit.draw(style=style, scale=0.7))

__all__ = ["HybridFCLSamplerQuantum"]
