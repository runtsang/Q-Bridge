"""Quantum sampler network with a 2‑qubit variational circuit.

The model mirrors the classical SamplerQNNGen but uses a parameterised
quantum circuit.  It exposes a `predict` method that returns the
probability distribution over the computational basis states.

Features
--------
* 3 layers of Ry rotations and CX entanglement
* Parameter vector split into input and weight parameters
* Uses Qiskit's StatevectorSampler for exact probability evaluation
* Can be used with the Qiskit Machine Learning SamplerQNN wrapper
"""
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class SamplerQNNGen:
    """
    Variational quantum sampler with 2 qubits and 4 trainable weight angles.
    """
    def __init__(self, input_dim: int = 2, hidden_layers: int = 3) -> None:
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.inputs = ParameterVector("input", self.input_dim)
        self.weights = ParameterVector("weight", 4 * self.hidden_layers)

        qc = QuantumCircuit(self.input_dim)

        # Layer 0: encode inputs
        for i in range(self.input_dim):
            qc.ry(self.inputs[i], i)

        # Variational layers
        for l in range(self.hidden_layers):
            base = l * 4
            qc.cx(0, 1)
            qc.ry(self.weights[base + 0], 0)
            qc.ry(self.weights[base + 1], 1)
            qc.cx(0, 1)
            qc.ry(self.weights[base + 2], 0)
            qc.ry(self.weights[base + 3], 1)

        self.circuit = qc

        # Sampler primitive
        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def predict(self, inputs: list[list[float]]) -> list[dict[str, float]]:
        """
        Evaluate the sampler for a batch of input angle pairs.

        Parameters
        ----------
        inputs : list of [theta0, theta1]

        Returns
        -------
        list of dicts mapping basis states to probabilities.
        """
        probs = self.sampler_qnn.predict(inputs)
        return probs

    def sample(self, inputs: list[list[float]], nshots: int = 1024) -> list[dict[str, int]]:
        """
        Draw samples from the quantum circuit.

        Parameters
        ----------
        inputs : list of [theta0, theta1]
        nshots : number of shots for each input

        Returns
        -------
        list of dicts mapping basis states to counts.
        """
        return self.sampler_qnn.sample(inputs, nshots=nshots)
