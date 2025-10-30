import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

class SamplerQNN_Gen318:
    """
    Quantum sampler that maps a 2‑dimensional input vector to a probability
    distribution over two outcomes using a 2‑qubit parametrised circuit.
    The circuit consists of amplitude‑encoding of the input, a small
    variational layer and an entangling step.
    """
    def __init__(self, n_params: int = 4):
        # Circuits parameters
        self.input_params = ParameterVector("input", 2)
        self.weight_params = ParameterVector("weight", n_params)

        # Build the template circuit
        self.qc = QuantumCircuit(2)
        # Amplitude encoding of the two‑dimensional input
        self.qc.ry(self.input_params[0], 0)
        self.qc.ry(self.input_params[1], 1)
        # Variational layer
        for i in range(n_params):
            self.qc.ry(self.weight_params[i], i % 2)
        # Entangling
        self.qc.cx(0, 1)
        self.qc.cx(1, 0)

        # Backend for statevector simulation
        self.sim = AerSimulator(method="statevector")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray
            Batch of 2‑dimensional input vectors of shape (batch, 2).

        Returns
        -------
        probs : np.ndarray
            Shape (batch, 2) containing the probability of measuring
            the first qubit in state |0⟩ or |1⟩.
        """
        probs = []
        for inp in x:
            bound_qc = self.qc.bind_parameters(
                {self.input_params[0]: inp[0], self.input_params[1]: inp[1]}
            )
            # Randomly initialise variational weights for demonstration
            weight_bind = {self.weight_params[i]: np.random.uniform(0, 2 * np.pi) for i in range(len(self.weight_params))}
            bound_qc = bound_qc.bind_parameters(weight_bind)

            state = Statevector(bound_qc)
            prob_dict = state.probabilities_dict()
            prob0 = prob_dict.get("00", 0.0) + prob_dict.get("01", 0.0)
            prob1 = prob_dict.get("10", 0.0) + prob_dict.get("11", 0.0)
            probs.append([prob0, prob1])
        return np.array(probs)
