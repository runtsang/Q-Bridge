from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class SamplerQNNHybridQML:
    """
    Quantum implementation of the hybrid sampler.
    Mirrors the classical QuantumFeatureExtractor but uses a real
    parameterised 2‑qubit circuit.  The sampler is built with
    StatevectorSampler for exact probability evaluation.
    """
    def __init__(self, clip: bool = True):
        self.clip = clip
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

        if self.clip:
            self._apply_clip()

        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=self.inputs,
                              weight_params=self.weights,
                              sampler=self.sampler)

    def _apply_clip(self):
        # Placeholder for parameter clipping logic
        pass

    def evaluate(self, input_vals: list[list[float]], weight_vals: list[list[float]]):
        """
        Evaluate the sampler for given input and weight values.
        input_vals: list of 2‑element lists
        weight_vals: list of 4‑element lists
        Returns a probability matrix (batch, 4 outcomes).
        """
        param_dict = {str(p): v for p, v in zip(self.inputs, input_vals)}
        param_dict.update({str(p): v for p, v in zip(self.weights, weight_vals)})
        return self.qnn.eval(param_dict)

__all__ = ["SamplerQNNHybridQML"]
