import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

class FCL:
    """
    Quantum implementation of a fully‑connected layer that uses a
    parameterised sampler circuit.  The class accepts a single vector of
    parameters ``thetas``:

        * the first two values are interpreted as input angles,
        * the next four values are treated as weight angles for the
          sampler circuit.

    The circuit is executed on a state‑vector sampler and the
    expectation value of the computational basis is returned.
    """
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Parameter vectors
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build the sampler circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)

        # SamplerQNN wrapper
        self.sampler = QSamplerQNN(
            circuit=qc,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=StatevectorSampler()
        )

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the sampler circuit with the supplied parameters.

        Parameters
        ----------
        thetas : array‑like
            Length‑6 array where the first two entries are input angles
            and the last four are weight angles.

        Returns
        -------
        expectation : np.ndarray
            One‑dimensional array containing the expectation value of the
            computational basis measurement.
        """
        if len(thetas)!= 6:
            raise ValueError("Expecting 6 parameters: 2 inputs + 4 weights.")

        # Bind parameters
        param_bind = {
            self.inputs[0]: thetas[0],
            self.inputs[1]: thetas[1],
            self.weights[0]: thetas[2],
            self.weights[1]: thetas[3],
            self.weights[2]: thetas[4],
            self.weights[3]: thetas[5],
        }

        # Sample
        result = self.sampler.run(param_bind)

        # Convert to expectation
        probs = result.sample_counts
        expectation = 0.0
        for state, count in probs.items():
            prob = count / self.shots
            expectation += int(state, 2) * prob
        return np.array([expectation])

    def get_sampler_parameters(self, thetas: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper that returns the weight part of ``thetas``.
        """
        return thetas[2:6]

__all__ = ["FCL"]
