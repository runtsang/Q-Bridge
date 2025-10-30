"""Hybrid quantum model combining a fully‑connected quantum layer and a two‑qubit sampler.

The circuit consists of a single‑qubit parameterised rotation for the fully‑connected
layer followed by a two‑qubit sampler circuit.  The sampler uses a StatevectorSampler
to evaluate the probability distribution, returning a single expectation value
that can be used as a classical loss term."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler

class HybridFCLSampler:
    def __init__(self, shots=100):
        # Parameters
        self.theta = Parameter("theta")
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build circuit
        self.circuit = QuantumCircuit(3)
        # Fully connected layer on qubit 0
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        # Sampler part on qubits 1 & 2
        self.circuit.ry(self.inputs[0], 1)
        self.circuit.ry(self.inputs[1], 2)
        self.circuit.cx(1, 2)
        self.circuit.ry(self.weights[0], 1)
        self.circuit.ry(self.weights[1], 2)
        self.circuit.cx(1, 2)
        self.circuit.ry(self.weights[2], 1)
        self.circuit.ry(self.weights[3], 2)
        self.circuit.measure_all()

        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.sampler = StatevectorSampler(self.backend)

    def run(self, thetas):
        """Run the hybrid circuit for a batch of theta values.

        Parameters
        ----------
        thetas : Iterable[float]
            Sequence of rotation angles for the fully connected layer.

        Returns
        -------
        np.ndarray
            Expectation values of the measurement outcomes.
        """
        # Bind parameters
        param_binds = [{self.theta: theta} for theta in thetas]
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridFCLSampler"]
