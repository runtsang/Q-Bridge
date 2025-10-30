import numpy as np
import torch

from qiskit.primitives import Sampler as QSampler
# Quantum helpers from the seed project
from SamplerQNN import SamplerQNN as QuantumSampler
from QCNN import QCNN as QuantumQCNN
from FCL import FCL as QuantumFCL

class SamplerQNN__gen093:
    """
    Hybrid quantum sampler network that combines a QCNN feature map,
    a fully‑connected quantum layer, and a sampler for probability
    estimation.  The run method evaluates the full quantum circuit
    and returns an expectation value; the sample method uses the
    quantum sampler directly.
    """

    def __init__(self) -> None:
        # Quantum components
        self.qcnn = QuantumQCNN()
        self.fcl = QuantumFCL()
        self.quantum_sampler = QuantumSampler()
        self.sampler = QSampler()

    def run(self, params: list[float]) -> np.ndarray:
        """
        Evaluate the combined quantum network.

        Parameters
        ----------
        params
            List of parameters for the fully‑connected layer circuit.

        Returns
        -------
        np.ndarray
            Expectation value of the measurement on the final qubit.
        """
        # Feature circuit from QCNN
        feature_circuit = self.qcnn()
        # Fully‑connected layer circuit
        fcl_circuit = self.fcl()
        # Bind the provided parameter to the single theta of the FCL circuit
        bound_fcl = fcl_circuit._circuit.bind_parameters({fcl_circuit.theta: params[0]})
        # Concatenate the two circuits
        combined = feature_circuit + bound_fcl
        # Sample using the primitive
        result = self.sampler.run(combined, shots=1024).result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / 1024.0
        expectation = np.sum(np.array(list(counts.keys())).astype(float) * probs)
        return np.array([expectation])

    def sample(self, params: list[float]) -> np.ndarray:
        """
        Run the quantum sampler circuit with custom parameters.

        Parameters
        ----------
        params
            List of parameters for the quantum circuit.

        Returns
        -------
        np.ndarray
            Probability distribution over measurement outcomes.
        """
        return self.quantum_sampler.sample(params)

__all__ = ["SamplerQNN__gen093"]
