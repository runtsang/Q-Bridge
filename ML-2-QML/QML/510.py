from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler
import numpy as np

class AdvancedSamplerQNN:
    """
    Quantum variational sampler that mirrors the classical architecture.
    Supports state‑vector or qasm backends, parameter‑shift gradient estimation,
    and explicit input binding.
    """

    def __init__(self, backend=None, shots=1024):
        """
        Parameters
        ----------
        backend : qiskit.providers.Backend, optional
            Backend for circuit execution. Defaults to the state‑vector simulator.
        shots : int
            Number of shots for sampling when using a qasm backend.
        """
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.shots = shots
        self._build_circuit()

    def _build_circuit(self):
        # Input parameters
        self.input_params = ParameterVector("x", 2)
        # Trainable variational weights
        self.weight_params = ParameterVector("w", 4)

        qc = QuantumCircuit(2)
        # Encode inputs
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)
        qc.cx(0, 1)
        # Two variational layers
        for i in range(2):
            qc.ry(self.weight_params[2 * i], 0)
            qc.ry(self.weight_params[2 * i + 1], 1)
            qc.cx(0, 1)

        self.circuit = qc
        self.sampler = StatevectorSampler(backend=self.backend)
        self.qnn = SamplerQNN(circuit=self.circuit,
                              input_params=self.input_params,
                              weight_params=self.weight_params,
                              sampler=self.sampler)

    def bind_inputs(self, inputs: np.ndarray):
        """
        Bind classical input values to the circuit parameters.
        """
        param_dict = {str(self.input_params[i]): float(v) for i, v in enumerate(inputs)}
        return self.qnn.bind_parameters(param_dict)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the sampler and return probabilities for the two basis states |00⟩ and |01⟩.
        """
        bound = self.bind_inputs(inputs)
        probs = bound.sample(num_shots=self.shots)
        probs_dict = probs.probabilities_dict()
        p0 = probs_dict.get("00", 0.0)
        p1 = probs_dict.get("01", 0.0)
        return np.array([p0, p1])

    def gradient(self, inputs: np.ndarray) -> np.ndarray:
        """
        Estimate gradients of the sampler parameters using the parameter‑shift rule.
        """
        bound = self.bind_inputs(inputs)
        return bound.gradient(self.weight_params)
